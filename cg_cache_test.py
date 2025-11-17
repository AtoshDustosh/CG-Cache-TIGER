import argparse
import json
import math
import pathlib
import pickle
import shutil
import time
import traceback

import numpy as np
import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader

from CHANGELOG import MODEL_VERSION
from tiger.data.data_loader import GraphCollator, load_jodie_data
from tiger.data.graph import Graph
from tiger.eval_utils import eval_edge_prediction, warmup
from tiger.model.feature_getter import NumericalFeature
from tiger.model.restarters import SeqRestarter, StaticRestarter
from tiger.model.tiger import TIGER
from tiger.utils import BackgroundThreadGenerator

from init_utils import init_data, init_model, init_parser
from train_utils import EarlyStopMonitor, get_logger, hash_args, seed_all

from torch.profiler import (
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)
import debug
import wandb as wb
import logging
import matplotlib.pyplot as plt
from pathlib import Path


def run(
    rank,
    world_size,
    *,
    prefix,
    root,
    data,
    dim,
    feature_as_buffer,
    gpu,
    seed,
    num_workers,
    subset,
    hit_type,
    restarter_type,
    hist_len,
    n_neighbors,
    n_layers,
    n_heads,
    dropout,
    strategy,
    msg_src,
    upd_src,
    mem_update_type,
    msg_tsfm_type,
    lr,
    n_epochs,
    bs,
    mutual_coef,
    patience,
    restart_prob,
    recover_from,
    recover_step,
    force,
    warmup_steps,
    cg_cache,
):

    # Get hash
    args = {
        k: v
        for k, v in locals().items()
        if not k in {"gpu", "force", "rank", "recover_from", "recover_step"}
    }
    HASH = hash_args(**args, MODEL_VERSION=MODEL_VERSION)
    prefix = HASH if prefix == "" else f"{prefix}.{HASH}"
    if gpu == "-1":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu}")

    # init logger
    logger = get_logger(HASH)
    logger.info(f"[START {HASH}]")
    logger.info(f"Model version: {MODEL_VERSION}")
    logger.info(", ".join([f"{k}={v}" for k, v in args.items()]))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Init
    seed_all(seed)
    # ============= Load Data ===========
    basic_data, graphs, dls, opt = init_data(
        data,
        root,
        seed,
        num_workers=num_workers,
        bs=bs,
        warmup_steps=warmup_steps,
        subset=subset,
        strategy=strategy,
        n_layers=n_layers,
        n_neighbors=n_neighbors,
        restarter_type=restarter_type,
        hist_len=hist_len,
        cg_cache=cg_cache,
        device=device,
    )
    (
        nfeats,
        efeats,
        full_data,
        train_data,
        val_data,
        test_data,
        inductive_val_data,
        inductive_test_data,
    ) = basic_data
    train_graph, full_graph = graphs
    (
        train_dl,
        offline_dl,
        val_dl,
        ind_val_dl,
        test_dl,
        ind_test_dl,
        val_warmup_dl,
        test_warmup_dl,
    ) = dls
    (
        train_collator,
        eval_collator,
    ) = opt

    # recover training
    epoch_start = 0

    epoch_times = []
    total_epoch_times = []

    for epoch in range(epoch_start, n_epochs):
        path_tracings = debug.profile_dir / f"data_only_epoch{epoch}"
        prof_ctx = (
            torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=debug.profiler_schedule,
                record_shapes=True,
                with_stack=True,
                on_trace_ready=tensorboard_trace_handler(str(path_tracings.absolute())),
            )
            if debug.ifprofile
            else debug.DummyProfiler()
        )
        with prof_ctx as prof:
            # Training
            start_epoch_t0 = time.time()
            logger.info("Start {} epoch".format(epoch))

            it = train_dl
            it = tqdm.tqdm(it, total=len(train_dl), ncols=50)

            batch_dumps = []

            for i_batch, (
                src_ids,
                dst_ids,
                neg_dst_ids,
                tss,
                eids,
                _,
                comp_graph,
            ) in enumerate(it):
                src_ids = src_ids.long().to(device)
                dst_ids = dst_ids.long().to(device)
                neg_dst_ids = neg_dst_ids.long().to(device)
                tss = tss.float().to(device)
                eids = eids.long().to(device)

                # Generate batch dumps
                if i_batch > 100:
                    break

                from tiger.data.data_classes import ComputationGraph

                layers = comp_graph.layers
                unique_ids = comp_graph.np_computation_graph_nodes
                unique_ids.sort()
                for i in range(len(layers)):
                    layers[i] = [
                        layers[i][0].cpu(),
                        layers[i][1].cpu(),
                        layers[i][2].cpu(),
                    ]
                    pass
                batch_dumps.append((layers, unique_ids))

                prof.step()
                pass

            path_dump = debug.profile_dir / f"cgs_{args["cg_cache"]}.pkl"

            with open(path_dump, "wb") as f:
                pickle.dump(batch_dumps, f)

        epoch_time = time.time() - start_epoch_t0
        epoch_times.append(epoch_time)
        train_collator.finish_epoch()

        total_epoch_time = time.time() - start_epoch_t0
        total_epoch_times.append(total_epoch_time)
        pass

    # Example data
    data = train_collator.cache_hits

    # Create the plot
    plt.figure(figsize=(12, 6))
    x = np.arange(1, len(data) + 1)
    plt.plot(
        x,
        data,
        label="Values",
        linestyle="-",
        color="tomato",
        linewidth=1,
    )
    plt.title("Cache hits")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save as image (PNG)
    plt.savefig(
        f"./tracings/cache_hit_{args["cg_cache"]}.png", dpi=300, bbox_inches="tight"
    )


def get_args():
    parser = init_parser()
    # Exp Setting
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix to name the checkpoints"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu", type=str, default="0", help="Cuda index")
    # Data
    parser.add_argument(
        "--subset", type=float, default=1.0, help="Only use a subset of training data"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers in train dataloader",
    )
    # Training
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--bs", type=int, default=200, help="Batch size")
    # MISC
    parser.add_argument(
        "--force", action="store_true", help="Overwirte the existing task"
    )
    parser.add_argument("--recover_from", type=str, default="", help="ckpt path")
    parser.add_argument("--recover_step", type=int, default=0, help="recover step")

    # MARKER Arguments for testing
    parser.add_argument(
        "--profile", action="store_true", default=False, help="Whether to profile"
    )
    parser.add_argument(
        "--cg_cache",
        type=float,
        default=0.0,
        help="If not 0.0, use sample cache with the given number as cache ratio to accelerate cg sampling.",
    )
    parser.add_argument(
        "--profile_dir",
        type=str,
        default="./tracings",
        help="The directory to store tracings.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    debug.setup(args)

    run(
        0,
        1,
        prefix=args.prefix,
        root=args.root,
        data=args.data,
        subset=args.subset,
        hit_type=args.hit_type,
        dim=args.dim,
        feature_as_buffer=not args.no_feat_buffer,
        gpu=args.gpu,
        seed=args.seed,
        num_workers=args.num_workers,
        restarter_type=args.restarter_type,
        hist_len=args.hist_len,
        n_neighbors=args.n_neighbors,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        strategy=args.strategy,
        msg_src=args.msg_src,
        upd_src=args.upd_src,
        mem_update_type=args.upd_fn,
        msg_tsfm_type=args.tsfm_fn,
        lr=args.lr,
        n_epochs=args.n_epochs,
        bs=args.bs,
        mutual_coef=args.mutual_coef,
        patience=args.patience,
        restart_prob=args.restart_prob,
        recover_from=args.recover_from,
        recover_step=args.recover_step,
        force=args.force,
        warmup_steps=args.warmup,
        cg_cache=args.cg_cache,
    )

    debug.run.finish()

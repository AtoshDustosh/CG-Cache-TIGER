from attr import dataclass
import torch
import numpy as np
import random

import wandb
import warnings
import functools
from torch.profiler import (
    ProfilerActivity,
    tensorboard_trace_handler,
    schedule,
)
from pathlib import Path


def deprecated(reason="This function is deprecated."):
    """
    A customized decorator.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapped

    return decorator


class DummyProfiler:
    def step(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# ---------------------------- Global Objects ----------------------------
ifprofile: bool
run: wandb.Run


def _setup_reproducibility(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pass


def _setup_wandb(args):
    global run
    run = wandb.init(
        group="global",
    )
    run.config.update(args)


profiler_schedule = schedule(
    wait=100,
    warmup=10,
    active=2,
    repeat=10,
    skip_first=20,
    skip_first_wait=1,
)
profile_dir = Path()


def _setup_profiler(args):
    global ifprofile, profiler_ctx, profiler_schedule, profile_dir
    ifprofile = args.profile

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)


def setup(args):
    _setup_reproducibility(args)
    _setup_wandb(args)
    _setup_profiler(args)


cache_hits = []


import socket
import pickle
import time
import struct

sender = socket.socket()
sender.connect(("192.168.1.102", 60000))


def send_obj_limited(sock, obj, bandwidth_MBps=1000, chunk_size=128 * 1024):
    """
    bandwidth_MBps: 限制带宽 (MB/s)
    chunk_size: 每次发送的块大小
    """
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    size = len(data)

    sock.sendall(struct.pack("!Q", size))

    bytes_per_sec = bandwidth_MBps * 1024 * 1024
    sleep_time = chunk_size / bytes_per_sec

    sent = 0
    while sent < size:
        end = min(sent + chunk_size, size)
        sock.sendall(data[sent:end])
        sent = end
        time.sleep(sleep_time)


ACK = b"\x01"


def send_and_wait_ack(sock, obj):
    t0 = time.time()

    send_obj_limited(sock, obj)  # 你的 pickle + 带宽受限版本
    ack = sock.recv(1)
    if ack == b"":
        raise RuntimeError("connection closed before ACK")
    assert ack == ACK, ack

    t1 = time.time()
    return t1 - t0

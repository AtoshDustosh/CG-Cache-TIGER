#!/bin/bash


export CUDA_VISIBLE_DEVICES=2

cache_rate=0.000
runtype="cache_rate_${cache_rate}"
profile_dir="tracings/train_${runtype}"
python train_self_supervised.py --data wikipedia \
    --n_layers 2 --n_epochs 3 \
    --profile --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}"

cache_rate=0.005
# ----------- cache 0.005 -----------
runtype="cache_rate_${cache_rate}"
profile_dir="tracings/train_${runtype}"
python train_self_supervised.py --data wikipedia \
    --n_layers 2 --n_epochs 3 \
    --profile --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}"


runtype="cache_rate_${cache_rate}"
profile_dir="tracings/train_${runtype}_async"
python train_self_supervised.py --data wikipedia \
    --n_layers 2 --n_epochs 3 \
    --profile --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}" \
    --async_cache


runtype="cache_rate_${cache_rate}"
profile_dir="tracings/train_${runtype}_redoNS"
python train_self_supervised.py --data wikipedia \
    --n_layers 2 --n_epochs 3 \
    --profile --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}" \
    --redo_NS


runtype="cache_rate_${cache_rate}"
profile_dir="tracings/train_${runtype}_async_redoNS"
python train_self_supervised.py --data wikipedia \
    --n_layers 2 --n_epochs 3 \
    --profile --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}" \
    --async_cache \
    --redo_NS



cache_rate=0.05
# ----------- cache 0.05 -----------
runtype="cache_rate_${cache_rate}"
profile_dir="tracings/train_${runtype}"
python train_self_supervised.py --data wikipedia \
    --n_layers 2 --n_epochs 3 \
    --profile --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}"


runtype="cache_rate_${cache_rate}"
profile_dir="tracings/train_${runtype}_async"
python train_self_supervised.py --data wikipedia \
    --n_layers 2 --n_epochs 3 \
    --profile --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}" \
    --async_cache


runtype="cache_rate_${cache_rate}"
profile_dir="tracings/train_${runtype}_redoNS"
python train_self_supervised.py --data wikipedia \
    --n_layers 2 --n_epochs 3 \
    --profile --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}" \
    --redo_NS


runtype="cache_rate_${cache_rate}"
profile_dir="tracings/train_${runtype}_async_redoNS"
python train_self_supervised.py --data wikipedia \
    --n_layers 2 --n_epochs 3 \
    --profile --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}" \
    --async_cache \
    --redo_NS
     
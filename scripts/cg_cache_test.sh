#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

cache_rate=0.05

runtype="cache_rate_${cache_rate}"

profile_dir="tracings/test_${runtype}"
rm -rf "${profile_dir}"

python cg_cache_test.py \
    --data wikipedia \
    --n_layers 2 \
    --n_epochs 1 \
    --profile \
    --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}" \
    --bs 200
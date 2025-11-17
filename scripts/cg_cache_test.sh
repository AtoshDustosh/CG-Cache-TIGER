#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

cache_rate=0.0005

if (( $(echo "$cache_rate > 0" | bc -l) )); then
    runtype="cached_${cache_rate}"
else
    runtype="uncached"
fi

profile_dir="tracings/test_${runtype}"
rm -rf "${profile_dir}"

python cg_cache_test.py \
    --data wikipedia \
    --n_layers 2 \
    --n_epochs 3 \
    --profile \
    --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}" \
    --bs 3
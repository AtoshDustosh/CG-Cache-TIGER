#!/bin/bash

rm -d -r results/* saved_models/* tracings/*

export CUDA_VISIBLE_DEVICES=3

python train_self_supervised.py --data wikipedia --n_layers 2 --n_epochs 3 --profile --cg_cache 0.05
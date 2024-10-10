#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

args=$(cat $1)

CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py ${args} 



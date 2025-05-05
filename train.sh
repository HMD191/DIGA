#!/bin/bash

# Step 1: Set environment variables
export UDATADIR="/kaggle/working/data"
export UPRJDIR="/kaggle/working/code"
export UOUTDIR="/kaggle/working/output"
export WANDB_API_KEY="c8703e1a9160e0b0af02b3a74b42d2c294bfbea3"
export NUM_WORKERS=1

# Step 2: Run training
python src/train.py \
    experiment=ttda \
    model/net=gta5_source \
    datamodule/test_list=cityscapes

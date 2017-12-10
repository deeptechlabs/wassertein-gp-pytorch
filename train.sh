#!/bin/bash
source activate dev3
python main.py --dataset='small-imagenet' --env_display=$RANDOM

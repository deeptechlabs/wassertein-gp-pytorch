#!/bin/sh
source activate dev3
python main.py --dataset='mnist'
python main.py --dataset='imagenet'

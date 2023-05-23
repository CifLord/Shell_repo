#!/bin/bash

echo starting
CUDA_VISIBLE_DEVICES=2 python run_predictions_gpu.py -i 'prediction/phase20003.lmdb'&
CUDA_VISIBLE_DEVICES=3 python run_predictions_gpu.py -i 'prediction/phase10002.lmdb'&

echo finished

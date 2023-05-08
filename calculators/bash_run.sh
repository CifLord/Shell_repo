#!/bin/bash

echo starting 

CUDA_VISIBLE_DEVICES=2 python run_predictions.py -m "mp-505009 " -i "prediction/phase1.lmdb" &
CUDA_VISIBLE_DEVICES=3 python run_predictions.py -m "candidates.json" -i "prediction/phase2.lmdb" &


echo finished

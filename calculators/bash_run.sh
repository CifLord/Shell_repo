#!/bin/bash

echo starting 

CUDA_VISIBLE_DEVICES=2 python run_predictions.py -m "Candidates/dgx_1.json " -i "prediction/phase1.lmdb" &
CUDA_VISIBLE_DEVICES=3 python run_predictions.py -m "Candidates/dgx_2.json" -i "prediction/phase2.lmdb" &


echo finished

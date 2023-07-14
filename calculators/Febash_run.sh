#!/bin/bash

echo starting 

CUDA_VISIBLE_DEVICES=2 python run_predictions.py -m "mp-18750 mp-38856" -i "prediction/FeMn.lmdb" &
CUDA_VISIBLE_DEVICES=3 python run_predictions.py -m "mp-640147 mp-22684" -i "prediction/FeNi.lmdb" &


echo finished

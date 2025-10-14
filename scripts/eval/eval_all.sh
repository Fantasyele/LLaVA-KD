#!/bin/bash

MODEL_PATH="Your_Model_Path"
MODEL_NAME="Model_Name"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/gqa.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/sqa.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=1 bash scripts/eval/textvqa.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=2 bash scripts/eval/pope.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=3 bash scripts/eval/mme.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=4 bash scripts/eval/mmmu.sh "$MODEL_PATH" "$MODEL_NAME" &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/vqav2.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=6 bash scripts/eval/vizwiz.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=7 bash scripts/eval/mmbench.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmbench_cn.sh "$MODEL_PATH" "$MODEL_NAME" &

wait        
#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib

MODEL_PATH="/mnt/data/LLaVA_KD/LLaVA_KD/pretrained_checkpoints/tiny-llava-Qwen2.5-1.5B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune"
MODEL_NAME="Qwen25_15B_DFT_250918_v1120"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/gqa.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/sqa.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=1 bash scripts/eval/textvqa.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=2 bash scripts/eval/pope.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=3 bash scripts/eval/mme.sh "$MODEL_PATH" "$MODEL_NAME" &
CUDA_VISIBLE_DEVICES=4 bash scripts/eval/mmmu.sh "$MODEL_PATH" "$MODEL_NAME" &

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/vqav2.sh "$MODEL_PATH" "$MODEL_NAME" &
# CUDA_VISIBLE_DEVICES=6 bash scripts/eval/vizwiz.sh "$MODEL_PATH" "$MODEL_NAME" &
# CUDA_VISIBLE_DEVICES=7 bash scripts/eval/mmbench.sh "$MODEL_PATH" "$MODEL_NAME" &
# CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmbench_cn.sh "$MODEL_PATH" "$MODEL_NAME" &

wait        
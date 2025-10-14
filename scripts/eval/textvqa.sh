#!/bin/bash


MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR="./eval_dataset/eval"

python3.12 -m llavakd.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python3.12 -m llavakd.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl


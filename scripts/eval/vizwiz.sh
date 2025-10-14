#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR="./eval_dataset/eval"

python3.12 -m llavakd.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/vizwiz/llava_test.jsonl \
    --image-folder $EVAL_DIR/vizwiz/test \
    --answers-file $EVAL_DIR/vizwiz/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python3.12 scripts/convert_vizwiz_for_submission.py \
    --annotation-file $EVAL_DIR/vizwiz/llava_test.jsonl \
    --result-file $EVAL_DIR/vizwiz/answers/$MODEL_NAME.jsonl \
    --result-upload-file $EVAL_DIR/vizwiz/answers_upload/$MODEL_NAME.json
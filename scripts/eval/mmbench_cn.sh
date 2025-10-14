#!/bin/bash

SPLIT="MMBench_DEV_CN_legacy"
MODEL_PATH=$1
MODEL_NAME=$2
<<<<<<< HEAD
EVAL_DIR="./eval_dataset/eval"

python3.12 -m llavakd.eval.model_vqa_mmbench \
=======
EVAL_DIR="./eval_dataset"

python -m llavakd.eval.model_vqa_mmbench \
>>>>>>> origin/main
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mmbench_cn/$SPLIT.tsv \
    --answers-file $EVAL_DIR/mmbench_cn/answers/$SPLIT/$MODEL_NAME.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi

mkdir -p $EVAL_DIR/mmbench_cn/answers_upload/$SPLIT

<<<<<<< HEAD
python3.12 scripts/convert_mmbench_for_submission.py \
=======
python scripts/convert_mmbench_for_submission.py \
>>>>>>> origin/main
    --annotation-file $EVAL_DIR/mmbench_cn/$SPLIT.tsv \
    --result-dir $EVAL_DIR/mmbench_cn/answers/$SPLIT \
    --upload-dir $EVAL_DIR/mmbench_cn/answers_upload/$SPLIT \
    --experiment $MODEL_NAME
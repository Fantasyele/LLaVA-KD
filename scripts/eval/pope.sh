#!/bin/bash


MODEL_PATH=$1
MODEL_NAME=$2
<<<<<<< HEAD
EVAL_DIR="./eval_dataset/eval"

python3.12 -m llavakd.eval.model_vqa_pope \
=======
EVAL_DIR="./eval_dataset"

python -m llavakd.eval.model_vqa_pope \
>>>>>>> origin/main
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

<<<<<<< HEAD
python3.12 llavakd/eval/eval_pope.py \
=======
python llavakd/eval/eval_pope.py \
>>>>>>> origin/main
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl

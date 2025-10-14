#!/bin/bash


MODEL_PATH=$1
MODEL_NAME=$2
<<<<<<< HEAD
EVAL_DIR="./eval_dataset/eval"

python3.12 -m llavakd.eval.model_vqa_science \
=======
EVAL_DIR="./eval_dataset"

python -m llavakd.eval.model_vqa_science \
>>>>>>> origin/main
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/scienceqa/llava_test_CQM-A.json \
    --image-folder $EVAL_DIR/scienceqa/images/test \
    --answers-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi

<<<<<<< HEAD
python3.12 llavakd/eval/eval_science_qa.py \
=======
python llavakd/eval/eval_science_qa.py \
>>>>>>> origin/main
    --base-dir $EVAL_DIR/scienceqa \
    --result-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --output-file $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_output.jsonl \
    --output-result $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_result.json


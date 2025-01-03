#!/bin/bash


MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR="./eval_dataset"  # If the evaluation fails, try changing the path to an absolute path

python -m llavakd.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MME/llava_mme.jsonl \
    --image-folder $EVAL_DIR/MME/MME_Benchmark_release_version \
    --answers-file $EVAL_DIR/MME/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
   --conv-mode phi

cd $EVAL_DIR/MME

python convert_answer_to_mme.py --experiment $MODEL_NAME

cd eval_tool

python calculation.py --results_dir answers/$MODEL_NAME


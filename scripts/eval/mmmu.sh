#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$2
<<<<<<< HEAD
EVAL_DIR="/mnt/data/LLaVA_KD/LLaVA_KD/eval_dataset/eval" # If the evaluation fails, try changing the path to an absolute path

python3.12 -m llavakd.eval.model_vqa_mmmu \
=======
EVAL_DIR="./eval_dataset"  # If the evaluation fails, try changing the path to an absolute path

python -m llavakd.eval.model_vqa_mmmu \
>>>>>>> origin/main
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MMMU/anns_for_eval.json \
    --image-folder $EVAL_DIR/MMMU/all_images \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

<<<<<<< HEAD
python3.12 scripts/convert_answer_to_mmmu.py \
=======
python scripts/convert_answer_to_mmmu.py \
>>>>>>> origin/main
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --answers-output $EVAL_DIR/MMMU/answers/${MODEL_NAME}_output.json

cd $EVAL_DIR/MMMU/eval

<<<<<<< HEAD
python3.12 main_eval_only.py --output_path $EVAL_DIR/MMMU/answers/${MODEL_NAME}_output.json
=======
python main_eval_only.py --output_path $EVAL_DIR/MMMU/answers/${MODEL_NAME}_output.json
>>>>>>> origin/main

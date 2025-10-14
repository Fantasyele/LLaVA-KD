

export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib

DATA_PATH=/mnt/data/mllm_datasets/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/mnt/data/mllm_datasets/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/mnt/data/mllm_datasets/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/mnt/data/mllm_datasets #finetune image dir

TEACHER_PATH=./teacher_pretrained_ckpt
TEACHER_NAME=tiny-llava-Qwen2-7B-siglip-so400m-patch14-384-base-
LLM_VERSION=pretrained_hg/Qwen2.5-0.5B # llm path in huggingface
VT_VERSION=pretrained_hg/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_base #chat template, other options are: phi, llama, gemmma, etc
VERSION=qwen2-0_5b_base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm

bash scripts/train/llava_kd_qwen2/finetune_distill_after_qwen2_sft.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$TEACHER_PATH" "$TEACHER_NAME"

DATA_PATH=./datasets/text_files/blip_laion_cc_sbu_558k.json
IMAGE_PATH=./datasets/llava/llava_pretrain/images #pretrain image dir

SHARE_PRETRAIN_DATA_PATH=./datasets/text_files/share-captioner_coco_lcs_sam_1246k_1107.json
SHARE_FINETUNE_DATA_PATH=./datasets/text_files/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json

SHARE_PRETRAIN_IMAGE_PATH=./datasets
SHARE_FINETUNE_IMAGE_PATH=./datasets


LLM_VERSION=pretrained_hg/phi-2
VT_VERSION=pretrained_hg/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=phi
VERSION=share
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=3072



# bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/share/pretrain_share.sh "$SHARE_PRETRAIN_DATA_PATH" "$SHARE_PRETRAIN_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" 
# bash scripts/train/share/finetune_share.sh "$SHARE_FINETUNE_DATA_PATH" "$SHARE_FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"

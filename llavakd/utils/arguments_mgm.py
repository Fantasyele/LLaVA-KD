from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, TYPE_CHECKING
import transformers


if TYPE_CHECKING:
    import transformers

@dataclass
class ModelArguments:
    # cache_dir: Optional[str] = field(default=None)

    model_name_or_path: Optional[str] = field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_aux: Optional[str] = field(default=None) # auxiliary vision tower
    optimize_vision_tower: bool = field(default=False) # whether to optimize vision tower
    optimize_vision_tower_aux: bool = field(default=False) # whether to optimize auxiliary vision tower
    drop_path: Optional[bool] = field(default=True) # whether to use drop path in auxiliary tower
    image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    # tokenizer_name_or_path: Optional[str] = field(default=None)
    # attn_implementation: Optional[str] = field(default=None)
    # mm_patch_merge_type: Optional[str] = field(default='flat')
    # resampler_hidden_size: Optional[int] = field(default=768)
    # num_queries: Optional[int] = field(default=128)
    # num_resampler_layers: Optional[int] = field(default=3)
    # model_max_length: int = field(
    #     default=512,
    #     metadata={
    #         "help":
    #             "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
    #     },
    # )
    # tokenizer_use_fast: bool = field(default=False)
    # tokenizer_padding_side: str = field(default='right')


@dataclass
class   DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_size_aux: Optional[int] = field(default=320)
    image_grid: Optional[int] = field(default=1)
    image_global: Optional[bool] = field(default=False)
    conv_version: str = 'pretrain'

@dataclass
class TrainingArguments(transformers.TrainingArguments):

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    lr_multi: Optional[str] = field(default=None)

    # training_recipe: str = field(default='common')
    # tune_type_llm: str = field(default="frozen") # support only: frozen, full, lora, qlora_int4, qlora_int8
    # tune_type_vision_tower: str = field(default="frozen") # support only: frozen, full, partially-tune
    # tune_vision_tower_from_layer: Optional[int] = field(default=10)
    # tune_type_connector: str = field(default="full") # support only: frozen, full
    # tune_embed_tokens: Optional[int] = field(default=False)
    # vision_tower_lr: Optional[float] = None
    # pretrained_model_path: Optional[str] = field(default=None)
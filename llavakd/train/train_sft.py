from packaging import version
import pathlib

import tokenizers
import transformers

import sys
sys.path.append('/mnt/data/LLaVA_KD/LLaVA_KD')
from llavakd.train.tinyllava_trainer import LLaVATrainer
from llavakd.training_recipe import TrainingRecipeFactory
from llavakd.utils import *
from llavakd.model import *
from llavakd.data.dataset import make_supervised_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments) 
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args

def get_conncet_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

def train():# 
    
    if os.path.exists(os.path.join('./checkpoints/qwen25_distill_llava_factory/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen2-0_5b_base-distill-pretrain', 'vision_tower')):
        pre_vision_tower_path = os.path.join('./checkpoints/qwen25_distill_llava_factory/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen2-0_5b_base-distill-pretrain', 'vision_tower/pytorch_model.bin')
        pre_vision = torch.load(pre_vision_tower_path, map_location='cpu')
        pre_connector_path = os.path.join('./checkpoints/qwen25_distill_llava_factory/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen2-0_5b_base-distill-pretrain', 'connector/pytorch_model.bin')
        pre_connector = torch.load(pre_connector_path, map_location='cpu')
        pre_language =  AutoModelForCausalLM.from_pretrained(os.path.join('./pretrained_hg', 'Qwen2.5-0.5B'), trust_remote_code=True)


    else:
        print('pretraining stage')
    
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    logger_setting(getattr(training_arguments, 'output_dir', None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = LLaVAKD(model_config)
    # load pretrained checkpoint
    if training_arguments.pretrained_model_path is not None:
        # model = training_recipe.load(model, model_args)

        # for qwen2
        model_config = TinyLlavaConfig.from_pretrained('./checkpoints/qwen25_distill_llava_factory/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen2-0_5b_base-distill-pretrain')
        model = LLaVAKD(model_config)
    
        # load from vision pretrained
        model.vision_tower._vision_tower.load_state_dict(pre_vision)
        print(f"Loading vision from {pre_vision_tower_path}")

        # load from connector pretrained
        model.connector._connector.load_state_dict(get_conncet_w(pre_connector, '_connector'))
        print(f"Loading connector from {pre_connector_path}")


        # load from pretrained
        for key, Value in pre_language.state_dict().items():
            model.language_model.state_dict()[key].copy_(Value)
        for key in pre_language.state_dict().keys():
            assert torch.equal(pre_language.state_dict()[key],
                            model.language_model.state_dict()[key]), f"Mismatch found in parameter: {key}"

        print(f"Loading language from Huggingface Pretrained Model")

    else:
        raise ValueError
    
    
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    log_trainable_params(model)  # not work well with zero3
    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)
    print('Training Start')
    trainer.train()
    
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    transformers.set_seed(1234)
    train()

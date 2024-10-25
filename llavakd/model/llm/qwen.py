from transformers import Qwen2ForCausalLM, AutoTokenizer
# from .modeling_qwen2 import Qwen2ForCausalLM
# from transformers import AutoTokenizer

from . import register_llm

@register_llm('qwen')
def return_phiclass():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.unk_token = tokenizer.pad_token
#        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return Qwen2ForCausalLM, (AutoTokenizer, tokenizer_and_post_load)

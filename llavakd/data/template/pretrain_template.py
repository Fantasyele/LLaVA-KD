from dataclasses import dataclass,field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import copy

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from ...utils.constants import *
from . import register_template

from transformers import PreTrainedTokenizer
import torch
    


@register_template('pretrain')
@dataclass
class PretrainTemplate(Template):
    # format_image_token: "Formatter" = EmptyFormatter(slot="")
    # format_user: "Formatter" = EmptyFormatter(slot="<image>")
    # format_assistant: "Formatter" = StringFormatter(slot="{{content}}\n")
    # system: "Formatter" = EmptyFormatter(slot="")
    # separator: "Formatter" = EmptyFormatter(slot=['', ''])
    
    format_image_token: "Formatter" = field(default_factory=lambda: EmptyFormatter(slot=""))
    format_user: "Formatter" = field(default_factory=lambda: EmptyFormatter(slot="<image>"))
    format_assistant: "Formatter" = field(default_factory=lambda: StringFormatter(slot="{{content}}\n"))
    system: "Formatter" = field(default_factory=lambda: EmptyFormatter(slot=""))
    separator: "Formatter" = field(default_factory=lambda: EmptyFormatter(slot=['', '']))

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        mask_len = len(self.tokenizer_image_token("<image>", tokenizer))
        labels[:mask_len] = IGNORE_INDEX
        return labels








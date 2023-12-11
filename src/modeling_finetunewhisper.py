import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel
from transformers import WhisperConfig, WhisperForConditionalGeneration

from transformers.deepspeed import is_deepspeed_zero3_enabled

try:
    from .configuration_finetunewhisper import FinetuneWhisperConfig
except:
    from configuration_finetunewhisper import FinetuneWhisperConfig

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

class FinetuneWhisperModel(PreTrainedModel):
    config_class = FinetuneWhisperConfig
    base_model_prefix = "finewhisper"

    def __init__(self, config: FinetuneWhisperConfig):
        super().__init__(config)
        self.whisper_config = WhisperConfig(**config.whisper_config)
        self.whisper_model = WhisperForConditionalGeneration(self.whisper_config)

    def forward(self, **kwargs):
        x =  self.whisper_model(**kwargs)
        return x
    
    def generate(self, **kwargs):
        return self.whisper_model.generate(**kwargs)
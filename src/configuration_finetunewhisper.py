"""BLSP config"""

from transformers import PretrainedConfig, WhisperConfig
from transformers import logging

logger = logging.get_logger(__name__)

class FinetuneWhisperConfig(PretrainedConfig):
    def __init__(
        self, 
        whisper_config=None, 
        freeze_whisper=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        if whisper_config is None:
            whisper_config = {}
            logger.info("whisper config is None. Initializing the WhisperConfig with default values")
        
        self.whisper_config = WhisperConfig(**whisper_config).to_dict()
        self.freeze_whisper = freeze_whisper
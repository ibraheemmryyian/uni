import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..utils.exceptions import ModelInitializationError
from ..config import MODEL_CONFIG
from ..logger import setup_logger

class LanguageModel:
    def __init__(self):
        self.logger = setup_logger(__name__, "logs/language_model.log")
        self.device = torch.device("cpu")
        
    def initialize(self):
        try:
            model_name = MODEL_CONFIG["language_model"]["name"]
            self.logger.info(f"Loading CPU-optimized {model_name}")
            
            # CPU-specific configuration
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            ).cpu()  # Force CPU placement
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.logger.info(f"{model_name} successfully loaded on CPU")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"CPU init failed: {str(e)}")
            raise ModelInitializationError(f"CPU model error: {str(e)}")
import logging
import torch
from transformers import AutoTokenizer
from llava.model import LlavaLlamaForCausalLM
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_name="llava-hf/llava-v1.6-mistral-7b-hf"):
    """Load LLaVA model and tokenizer"""
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        logger.info("Loading model...")
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    model, tokenizer = load_model()
    if model is not None:
        logger.info("Model loaded successfully!")
        device = next(model.parameters()).device
        logger.info(f"Model device: {device}")
    else:
        logger.error("Failed to load model")
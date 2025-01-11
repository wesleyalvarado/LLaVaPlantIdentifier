import os
import gc
import torch
import traceback
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoConfig
)
import logging
from torch.utils.data import DataLoader

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Import local modules
from data.dataset import MemoryEfficientPlantDataset, test_dataset
from models.trainer import CustomTrainer
from config.training_config import get_training_args, ModelConfig
from utils.logging_utils import setup_logging

def train_llava_model():
    """Main training function with enhanced error handling and dataset validation"""
    try:
        # Setup
        model_dir = os.path.expanduser("~/plant_models/llava_plant_model")
        os.makedirs(model_dir, exist_ok=True)
        offload_dir = os.path.join(model_dir, "offload")
        os.makedirs(offload_dir, exist_ok=True)
        
        setup_logging(os.path.join(model_dir, "logs"))
        logger = logging.getLogger(__name__)
        
        # Aggressive memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load configurations
        model_config = ModelConfig()
        logger.info(f"Loading model: {model_config.name}")
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_config.name,
            trust_remote_code=True
        )
        
        # Load model configuration
        config = AutoConfig.from_pretrained(
            model_config.name,
            trust_remote_code=True
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_config.name,
            config=config,
            torch_dtype=getattr(torch, model_config.dtype),
            device_map=model_config.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Prepare datasets
        train_dataset = MemoryEfficientPlantDataset(
            processor=processor,
            split="train",
            sample_fraction=0.1,
            image_size=model_config.image_size
        )
        
        eval_dataset = MemoryEfficientPlantDataset(
            processor=processor,
            split="test",
            sample_fraction=0.1,
            image_size=model_config.image_size
        )
        
        # Initialize trainer
        training_args = get_training_args(model_dir)
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        # Start training
        logger.info("Beginning model training...")
        trainer.train()
        
        # Save final model
        final_output_dir = os.path.join(model_dir, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        trainer.save_model(final_output_dir)
        processor.save_pretrained(final_output_dir)
        
        logger.info(f"Training completed. Model saved to {final_output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    train_llava_model()

# test_trainer.py
import os
import torch
import logging
from models.trainer import CustomTrainer
from data.dataset import MemoryEfficientPlantDataset
from config.training_config import get_training_args
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from huggingface_hub import login

# Setup detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_batch():
    """Test processing a single batch through the trainer"""
    try:
    # First authenticate with Hugging Face
        token = os.getenv("HUGGINGFACE_TOKEN")  # Fetch token from environment variable
        
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")
        
        login(token)
        
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", use_auth_token=token)
        
        logger.info("Processor loaded successfully.")
        
        logger.info("Loading model...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16
        )
        
        logger.info("Creating test dataset...")
        test_dataset = MemoryEfficientPlantDataset(
            split="train",
            sample_fraction=0.01  # Just use 1% of data
        )
        
        # Get a single batch
        logger.info("Getting single batch...")
        batch = test_dataset[0]
        logger.info(f"Batch keys: {batch.keys()}")
        
        # Print tensor shapes
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"{key} shape: {value.shape}")
            else:
                logger.info(f"{key}: {value}")
        
        # Setup trainer
        logger.info("Setting up trainer...")
        training_args = get_training_args("test_output")
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=test_dataset
        )
        
        # Process single batch
        logger.info("Processing batch through trainer...")
        try:
            loss = trainer.training_step(model, batch)
            logger.info(f"Successfully processed batch with loss: {loss}")
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    test_single_batch()
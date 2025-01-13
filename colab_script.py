import os
import gc
import torch
import logging
import traceback
import models.trainer
from models.trainer import CustomTrainer
from huggingface_hub import login
from transformers import (
    AutoProcessor,
    AutoModel,
    LlavaNextForConditionalGeneration
)
from config.training_config import ModelConfig
from utils.logging_utils import setup_logging
from utils.tokenizer_utils import smart_tokenizer_and_embedding_resize

# Global variables
global_model = None
global_processor = None

def load_llava_model():
    """
    Load the LLaVA-Next model and processor
    """
    global global_model, global_processor
    
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load configurations
        model_config = ModelConfig()
        logger.info(f"Loading model: {model_config.name}")
        
        # Load processor
        logger.info("Loading processor...")
        global_processor = AutoProcessor.from_pretrained(
            model_config.name,
            trust_remote_code=True
        )
        
        # Load model
        logger.info("Loading model...")
        global_model = LlavaNextForConditionalGeneration.from_pretrained(
            model_config.name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Add pad token if needed
        if global_processor.tokenizer.pad_token is None:
            logger.info("Adding pad token to tokenizer and resizing embeddings")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=global_processor.tokenizer,
                model=global_model,
            )
        
        # Move to GPU
        global_model.cuda()
        
        logger.info("Model loaded successfully")
        logger.info(f"Current GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        global_model = None
        global_processor = None
        torch.cuda.empty_cache()
        gc.collect()
        raise

def train_llava_model():
    """Train the loaded model"""
    global global_model, global_processor
    
    try:
        # Verify model and processor are loaded
        if global_model is None or global_processor is None:
            raise ValueError("Model or processor not loaded. Run load_llava_model() first!")
            
        # Setup
        setup_logging()
        logger = logging.getLogger(__name__)
        model_dir = os.path.expanduser("~/plant_models/llava_plant_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        # Prepare datasets with correct parameters
        train_dataset = MemoryEfficientPlantDataset(
            split="train",
            sample_fraction=0.1,
            image_size=ModelConfig().image_size
        )
        
        eval_dataset = MemoryEfficientPlantDataset(
            split="test",
            sample_fraction=0.1,
            image_size=ModelConfig().image_size
        )
        
        # Initialize trainer
        training_args = get_training_args(model_dir)
        trainer = CustomTrainer(
            model=global_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        # Start training
        logger.info("Beginning model training...")
        logger.info(f"Available methods: {dir(trainer)}")
        trainer.train()
        
        # Save final model
        final_output_dir = os.path.join(model_dir, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        trainer.save_model(final_output_dir)
        global_processor.save_pretrained(final_output_dir)
        
        logger.info(f"Training completed. Model saved to {final_output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

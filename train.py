import os
import gc
import torch
import traceback
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoConfig
)
import logging
from torch.utils.data import DataLoader

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

from data.dataset import MemoryEfficientPlantDataset, memory_efficient_collate_fn
from models.trainer import CustomTrainer
from config.training_config import get_training_args, ModelConfig
from utils.logging_utils import setup_logging

def validate_dataset(dataset, processor, logger, max_samples_to_check=10):
    """
    Validate dataset processing by attempting to process a subset of samples
    """
    logger.info(f"Validating dataset processing for {max_samples_to_check} samples...")
    
    # Create a DataLoader to test batch processing
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # Small batch size for detailed error tracking
        collate_fn=memory_efficient_collate_fn,
        num_workers=0,  # Disable multiprocessing for debugging
        pin_memory=False  # Disable pin memory for lower memory usage
    )
    
    successful_samples = 0
    failed_samples = []
    
    for idx, batch in enumerate(dataloader):
        if idx >= max_samples_to_check:
            break
            
        if batch is None:
            failed_samples.append(idx)
            logger.warning(f"Failed to process sample {idx}")
        else:
            successful_samples += 1
    
    success_rate = (successful_samples / (successful_samples + len(failed_samples))) * 100
    logger.info(f"Dataset validation results:")
    logger.info(f"  Successful samples: {successful_samples}")
    logger.info(f"  Failed samples: {len(failed_samples)}")
    logger.info(f"  Success rate: {success_rate:.2f}%")
    
    if success_rate < 50:
        logger.error("Low dataset processing success rate. Check image preprocessing.")
        raise ValueError("Dataset processing failed for too many samples")
    
    return successful_samples, failed_samples

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
            torch.cuda.reset_max_memory_allocated()
        
        # Load configurations
        model_config = ModelConfig()
        logger.info(f"Loading model: {model_config.name}")
        
        # Load processor
        processor = LlavaNextProcessor.from_pretrained(
            model_config.name,
            trust_remote_code=model_config.trust_remote_code
        )
        
        # Load model configuration and check components
        config = AutoConfig.from_pretrained(
            model_config.name,
            trust_remote_code=model_config.trust_remote_code
        )
        logger.debug(f"Vision config: {config.vision_config}")
        logger.debug(f"Text config: {config.text_config}")
        
        # Memory cleanup before model loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model with memory optimizations
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_config.name,
            config=config,
            torch_dtype=getattr(torch, model_config.dtype),
            low_cpu_mem_usage=True,
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
            max_memory={0: "18GB"},  # Explicit memory limit
            offload_folder=offload_dir
        )
        
        # Verify model components
        if not hasattr(model, 'vision_model') and not hasattr(model, 'vision_tower'):
            logger.error("Vision components not found")
            raise ValueError("Vision model components missing")
        
        # Selective parameter freezing
        for param in model.parameters():
            param.requires_grad = False
        
        if hasattr(model, 'language_model'):
        # For Mistral, unfreeze the final layers of the language model
            for name, param in model.language_model.named_parameters():
        # Unfreeze only the final layers
                if any(layer_name in name for layer_name in ['layer.31', 'norm']):
                    param.requires_grad = True
                    logger.debug(f"Unfrozen parameter: {name}")

        
        # Log parameter status
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Memory cleanup before dataset loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare datasets with comprehensive validation
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
        
        # Validate datasets before training
        logger.info("Validating train dataset...")
        train_successful, train_failed = validate_dataset(
            train_dataset, 
            processor, 
            logger,
            max_samples_to_check=10
        )
        
        logger.info("Validating eval dataset...")
        eval_successful, eval_failed = validate_dataset(
            eval_dataset, 
            processor, 
            logger,
            max_samples_to_check=5
        )
        
        # Memory cleanup before training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize trainer with memory optimizations
        training_args = get_training_args(model_dir)
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=memory_efficient_collate_fn
            )
        
        # Start training
        logger.info("Beginning model training...")
        trainer.train()
        
        # Memory cleanup before saving
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

if __name__ == "__main__":
    train_llava_model()
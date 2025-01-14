# test_trainer.py
import os
import torch
import logging
import psutil
import gc
import traceback
from huggingface_hub import login
from transformers import (
    AutoProcessor, 
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
    ProcessorConfig
)
from models.trainer import CustomTrainer
from data.dataset import MemoryEfficientPlantDataset
from config.training_config import get_training_args, ModelConfig

# Setup detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_processor_and_model(processor, model):
    """Configure processor and model with required attributes."""
    # Configure processor
    config = ProcessorConfig(
        patch_size=14,
        vision_feature_select_strategy='default',
        image_size=336
    )
    processor.config = config
    
    # Set patch_size directly
    processor.patch_size = 14
    
    # Set vision feature strategy
    processor.vision_feature_select_strategy = 'default'
    
    # Configure tokenizer padding
    processor.tokenizer.padding_side = 'right'
    model.padding_side = 'right'
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Verify configuration
    logger.info("\nProcessor configuration:")
    logger.info(f"  patch_size: {getattr(processor, 'patch_size', None)}")
    logger.info(f"  vision_feature_select_strategy: {getattr(processor, 'vision_feature_select_strategy', None)}")
    logger.info(f"  config.patch_size: {getattr(processor.config, 'patch_size', None)}")
    logger.info(f"  tokenizer padding_side: {processor.tokenizer.padding_side}")
    logger.info(f"  model padding_side: {model.padding_side}")
    
    return processor, model

def test_single_batch():
    """Test processing a single batch through the trainer"""
    try:
        # Clear CUDA cache and collect garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        # First authenticate with Hugging Face
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")
        login(token)
        
        try:
            # Set memory allocation strategy
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.7)
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
            
            logger.info("Loading model and processor...")
            model_config = ModelConfig()
            
            # Load processor first
            processor = AutoProcessor.from_pretrained(
                model_config.name,
                trust_remote_code=True
            )
            
            # Configure processor before model loading
            processor, _ = configure_processor_and_model(processor, None)
            logger.info("Processor configured")
            
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            # Load model with memory optimizations
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_config.name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                offload_folder="offload",
                low_cpu_mem_usage=True
            )
            
            # Configure model padding side
            _, model = configure_processor_and_model(processor, model)
            logger.info("Model configured")
            
            # Create test dataset
            logger.info("Creating test dataset...")
            test_dataset = MemoryEfficientPlantDataset(
                split="train",
                sample_fraction=0.01,
                processor=processor
            )
            
            # Get a single batch
            logger.info("Getting single batch...")
            batch = test_dataset[0]
            
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
                logger.error(traceback.format_exc())
                raise
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            raise
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    test_single_batch()
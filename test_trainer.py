# test_trainer.py
import os
import torch
import logging
import gc
import time
import psutil
import traceback
from huggingface_hub import login
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)
from models.trainer import CustomTrainer
from data.dataset import MemoryEfficientPlantDataset
from config.training_config import get_training_args, ModelConfig
from utils.tokenizer_utils import smart_tokenizer_and_embedding_resize

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_system_info():
    """Log system information for debugging."""
    logger.info("System Information:")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f}GB")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB")
    logger.info(f"PyTorch Version: {torch.__version__}")

def configure_processor_and_model(processor, model):
    """Configure processor with values from model config."""
    # Get values from model config
    patch_size = model.config.vision_config.patch_size
    vision_strategy = model.config.vision_feature_select_strategy
    image_size = model.config.vision_config.image_size
    
    # Configure vision processor
    if hasattr(processor, 'image_processor'):
        processor.image_processor.patch_size = patch_size
        processor.image_processor.vision_feature_select_strategy = vision_strategy
        processor.image_processor.size = {'height': image_size, 'width': image_size}
        if hasattr(processor.image_processor, 'config'):
            processor.image_processor.config.patch_size = patch_size
            processor.image_processor.config.image_size = image_size
            processor.image_processor.config.vision_feature_select_strategy = vision_strategy
    
    # Configure processor
    processor.patch_size = patch_size
    processor.vision_feature_select_strategy = vision_strategy
    processor.image_size = image_size
    
    if not hasattr(processor, 'config'):
        processor.config = type('ProcessorConfig', (), {})()
    processor.config.patch_size = patch_size
    processor.config.image_size = image_size
    processor.config.vision_feature_select_strategy = vision_strategy
    
    # Configure tokenizer
    processor.tokenizer.padding_side = 'right'
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Log configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Model config values:")
    logger.info(f"    patch_size: {patch_size}")
    logger.info(f"    vision_feature_select_strategy: {vision_strategy}")
    logger.info(f"    image_size: {image_size}")
    logger.info(f"  Processor values:")
    logger.info(f"    patch_size: {processor.patch_size}")
    logger.info(f"    vision_feature_select_strategy: {processor.vision_feature_select_strategy}")
    logger.info(f"    image_processor.patch_size: {getattr(processor.image_processor, 'patch_size', None)}")
    logger.info(f"    image_processor.vision_feature_select_strategy: {getattr(processor.image_processor, 'vision_feature_select_strategy', None)}")
    
    return processor, model

def configure_model(model):
    """Configure model with required attributes."""
    # Set padding side
    model.config.padding_side = 'right'
    model.padding_side = 'right'
    
    # Ensure correct image settings
    if hasattr(model.config, 'vision_config'):
        model.config.vision_config.image_size = 336
        model.config.vision_config.patch_size = 14
    
    logger.info("\nModel configuration:")
    logger.info(f"  padding_side: {model.padding_side}")
    logger.info(f"  config padding_side: {model.config.padding_side}")
    if hasattr(model.config, 'vision_config'):
        logger.info(f"  vision_config.image_size: {model.config.vision_config.image_size}")
        logger.info(f"  vision_config.patch_size: {model.config.vision_config.patch_size}")
    
    return model

def test_memory_usage():
    """Test memory usage during model operations."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # Run garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Log memory statistics
        current_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        logger.info("\nMemory Usage Statistics:")
        logger.info(f"  Initial Memory: {initial_memory / 1024**2:.2f}MB")
        logger.info(f"  Current Memory: {current_memory / 1024**2:.2f}MB")
        logger.info(f"  Peak Memory: {peak_memory / 1024**2:.2f}MB")
        
        return peak_memory < torch.cuda.get_device_properties(0).total_memory * 0.9
    return True

def test_single_batch():
    """Test processing a single batch through the trainer"""
    try:
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        # Set CUDA launch blocking for better error messages
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Authenticate with Hugging Face
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")
        login(token)
        
        # Get model config
        model_config = ModelConfig()

        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load model first
        logger.info("Loading model...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_config.name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load and configure processor
        logger.info("Loading and configuring processor...")
        processor = AutoProcessor.from_pretrained(
            model_config.name,
            trust_remote_code=True
        )
        
        # Configure processor and model
        processor, model = configure_processor_and_model(processor, model)
        logger.info("Processor and model configured")
        model = configure_model(model)
        
        # Create test dataset
        logger.info("Creating test dataset...")
        test_dataset = MemoryEfficientPlantDataset(
            processor=processor,
            split="train",
            sample_fraction=0.01
        )
        
        # Get single batch
        logger.info("Getting single batch...")
        batch = test_dataset[0]
        
        # Debug tensor shapes
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
        
        # Process batch
        logger.info("Processing batch through trainer...")
        loss = trainer.training_step(model, batch)
        logger.info(f"Successfully processed batch with loss: {loss}")
        
        return True
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_multiple_batches(num_batches=3):
    """Test processing multiple batches to verify stability."""
    try:
        logger.info(f"\nTesting {num_batches} batches for stability...")
        losses = []
        memory_usage = []
        
        for i in range(num_batches):
            start_time = time.time()
            
            # Record initial memory
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
            
            # Process batch
            success = test_single_batch()
            if not success:
                logger.error(f"Batch {i+1} failed")
                return False
            
            # Record memory after batch
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated()
                memory_diff = final_memory - initial_memory
                memory_usage.append(memory_diff)
            
            # Calculate batch time
            batch_time = time.time() - start_time
            logger.info(f"Batch {i+1} completed in {batch_time:.2f}s")
            
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Log memory statistics
        if torch.cuda.is_available():
            logger.info("\nMemory Usage Statistics:")
            logger.info(f"  Average memory difference: {sum(memory_usage) / len(memory_usage) / 1024**2:.2f}MB")
            logger.info(f"  Max memory difference: {max(memory_usage) / 1024**2:.2f}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Multiple batch test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests."""
    try:
        # Log system information
        log_system_info()
        
        # Run memory test
        logger.info("\nRunning memory usage test...")
        if not test_memory_usage():
            logger.error("Memory usage test failed")
            return False
        
        # Run single batch test
        logger.info("\nRunning single batch test...")
        if not test_single_batch():
            logger.error("Single batch test failed")
            return False
        
        # Run multiple batch test
        logger.info("\nRunning multiple batch test...")
        if not test_multiple_batches(num_batches=3):
            logger.error("Multiple batch test failed")
            return False
        
        logger.info("\nAll tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main()
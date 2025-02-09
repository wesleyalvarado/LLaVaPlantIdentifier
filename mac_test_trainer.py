# mac_test_trainer.py

import os
import torch
import logging
import gc
import time
import sys
import psutil
import traceback
from pathlib import Path
from huggingface_hub import login
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    AutoConfig
)
from models.mac_trainer import CustomTrainer
from data.mac_dataset import MemoryEfficientPlantDataset
from config.mac_training_config import (
    get_training_args,
    ModelConfig,
    OptimizationConfig,
    DataConfig,
    get_device
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_system_info():
    """Log system information for debugging."""
    try:
        logger.info("\nSystem Information:")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CPU Count: {psutil.cpu_count()}")
        logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f}GB")
        logger.info(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.2f}GB")
        logger.info(f"MPS Available: {torch.backends.mps.is_available()}")
        
    except Exception as e:
        logger.error(f"Error logging system info: {e}")

def setup_test_environment():
    """Setup test environment including output directories."""
    try:
        test_dir = Path("test_output")
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "checkpoints").mkdir(exist_ok=True)
        (test_dir / "cache").mkdir(exist_ok=True)
        (test_dir / "logs").mkdir(exist_ok=True)
        return test_dir
    except Exception as e:
        logger.error(f"Error setting up test environment: {e}")
        raise

def cleanup_test_environment():
    """Clean up test environment and memory."""
    try:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        logger.error(f"Error in cleanup: {e}")

def load_model_for_testing(model_config: ModelConfig, device):
    """Load model with proper type configuration."""
    try:
        logger.info(f"Loading model using device: {device}")
        
        # Load model with float32 as base type
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_config.name,
            torch_dtype=torch.float32,  # Changed from float16
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Convert model parameters to float32
        model = model.to(dtype=torch.float32)
        
        # Move model to device
        model = model.to(device)
        
        # Enable memory optimization
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
            
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return None

def setup_processor(model_config: ModelConfig):
    """Setup and configure processor."""
    try:
        processor = AutoProcessor.from_pretrained(
            model_config.name,
            trust_remote_code=True
        )
        
        # Configure processor
        processor.image_processor.size = {
            'height': model_config.image_size,
            'width': model_config.image_size
        }
        processor.image_processor.patch_size = model_config.patch_size
        processor.tokenizer.padding_side = 'right'
        
        return processor
    except Exception as e:
        logger.error(f"Error setting up processor: {e}")
        logger.error(traceback.format_exc())
        return None

def test_model_loading(device):
    """Test model loading and configuration."""
    try:
        logger.info("Testing model loading...")
        model_config = ModelConfig()
        model = load_model_for_testing(model_config, device)
        
        if model is None:
            return False
        
        # Verify model is on correct device
        sample_param = next(model.parameters())
        logger.info(f"Model parameter device: {sample_param.device}")
        
        # Test basic forward pass
        logger.info("Testing basic forward pass...")
        dummy_input = torch.ones(1, 3, 336, 336, device=device)
        try:
            with torch.no_grad():
                _ = model.vision_tower(dummy_input)
            logger.info("Basic forward pass successful")
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        return False

def test_processor_setup():
    """Test processor setup and configuration."""
    try:
        logger.info("Testing processor setup...")
        model_config = ModelConfig()
        processor = setup_processor(model_config)
        
        if processor is None:
            return False
        
        # Verify processor configuration
        assert processor.image_processor.size['height'] == model_config.image_size
        assert processor.image_processor.patch_size == model_config.patch_size
        assert processor.tokenizer.padding_side == 'right'
        
        logger.info("Processor configuration verified")
        return True
    except Exception as e:
        logger.error(f"Processor setup test failed: {e}")
        return False

def test_dataset_creation(processor, device):
    """Test dataset creation and processing."""
    try:
        logger.info("Testing dataset creation...")
        dataset = MemoryEfficientPlantDataset(
            processor=processor,
            split="train",
            sample_fraction=0.02,
            cache_dir="test_output/cache",
            device=device,
            batch_size=2
        )
        
        if len(dataset) == 0:
            logger.error("Dataset is empty")
            return False
        
        # Test single item loading
        item = dataset[0]
        required_keys = ['pixel_values', 'input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key not in item:
                logger.error(f"Missing required key: {key}")
                return False
            if not isinstance(item[key], torch.Tensor):
                logger.error(f"Key {key} is not a tensor")
                return False
        
        logger.info(f"Created dataset with {len(dataset)} samples")
        return True
    except Exception as e:
        logger.error(f"Dataset creation test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_single_batch(model, processor, device):
    """Test processing a single batch."""
    try:
        logger.info("Testing single batch processing...")
        # Create test dataset
        dataset = MemoryEfficientPlantDataset(
            processor=processor,
            split="train",
            sample_fraction=0.02,
            cache_dir="test_output/cache",
            device=device,
            batch_size=2
        )
        
        # Get single batch
        batch = dataset[0]
        
        # Setup trainer
        training_args = get_training_args(
            "test_output",
            ModelConfig(),
            OptimizationConfig(),
            DataConfig()
        )
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )
        
        # Process batch
        loss = trainer.training_step(model, batch)
        if loss is None:
            logger.error("Training step returned None")
            return False
        
        logger.info(f"Batch processed with loss: {loss.item():.4f}")
        return True
    except Exception as e:
        logger.error(f"Single batch test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_multiple_batches(model, processor, device, num_batches=2):
    """Test processing multiple batches."""
    try:
        logger.info(f"Testing {num_batches} batches...")
        
        # Create dataset
        dataset = MemoryEfficientPlantDataset(
            processor=processor,
            split="train",
            sample_fraction=0.02,
            cache_dir="test_output/cache",
            device=device,
            batch_size=2
        )
        
        # Setup trainer
        training_args = get_training_args(
            "test_output",
            ModelConfig(),
            OptimizationConfig(),
            DataConfig()
        )
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )
        
        # Process batches
        for i in range(num_batches):
            batch = dataset[i]
            loss = trainer.training_step(model, batch)
            
            if loss is None:
                logger.error(f"Batch {i+1} failed")
                return False
            
            logger.info(f"Batch {i+1} completed with loss: {loss.item():.4f}")
            cleanup_test_environment()
        
        return True
    except Exception as e:
        logger.error(f"Multiple batch test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_all_tests():
    """Run all tests in sequence."""
    try:
        # Setup test environment
        test_dir = setup_test_environment()
        
        # Log system information
        log_system_info()
        
        # Get device
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Load model configuration
        model_config = ModelConfig()
        
        # Load model and processor
        model = load_model_for_testing(model_config, device)
        processor = setup_processor(model_config)
        
        if None in (model, processor):
            logger.error("Failed to initialize model or processor")
            return False
        
        tests = [
            ("Model Loading", lambda: test_model_loading(device)),
            ("Processor Setup", test_processor_setup),
            ("Dataset Creation", lambda: test_dataset_creation(processor, device)),
            ("Single Batch", lambda: test_single_batch(model, processor, device)),
            ("Multiple Batches", lambda: test_multiple_batches(model, processor, device, 2))
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\nRunning {test_name} test...")
            try:
                start_time = time.time()
                success = test_func()
                duration = time.time() - start_time
                results.append((test_name, success, duration))
                logger.info(f"{test_name} test: {'Passed' if success else 'Failed'} in {duration:.2f}s")
            except Exception as e:
                logger.error(f"{test_name} test failed with error: {e}")
                results.append((test_name, False, 0))
            
            cleanup_test_environment()
        
        # Log final results
        logger.info("\nTest Results:")
        for test_name, success, duration in results:
            logger.info(f"{test_name}: {'Passed' if success else 'Failed'} ({duration:.2f}s)")
        
        return all(success for _, success, _ in results)
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        cleanup_test_environment()

if __name__ == "__main__":
    # Check for HUGGINGFACE_TOKEN
    if not os.getenv("HUGGINGFACE_TOKEN"):
        logger.error("HUGGINGFACE_TOKEN environment variable not set")
        exit(1)
        
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    success = run_all_tests()
    exit(0 if success else 1)
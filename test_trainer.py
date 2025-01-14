# test_trainer.py
import os
import torch
import logging
import gc
import time
import psutil
import traceback
from pathlib import Path
from huggingface_hub import login
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)
from models.trainer import CustomTrainer
from data.dataset import MemoryEfficientPlantDataset
from config.training_config import get_training_args, ModelConfig, OptimizationConfig, DataConfig
from utils.tokenizer_utils import smart_tokenizer_and_embedding_resize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_system_info():
    """Log system information for debugging."""
    logger.info("\nSystem Information:")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f}GB")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB")
    logger.info(f"PyTorch Version: {torch.__version__}")

def setup_test_environment():
    """Setup test environment including output directories."""
    test_dir = Path("test_output")
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "checkpoints").mkdir(exist_ok=True)
    (test_dir / "cache").mkdir(exist_ok=True)
    return test_dir

def cleanup_test_environment(test_dir: Path):
    """Clean up test environment."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_model_for_testing(model_config: ModelConfig):
    """Load model with appropriate configuration for testing."""
    try:
        # First, fix the missing import
        from transformers import AutoConfig
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        # Create comprehensive device map including vision tower components
        device_map = {
            'model.embed_tokens': 'cuda:0',
            'model.layers': 'cuda:0',
            'lm_head': 'cpu',
            'model.norm': 'cuda:0',
            'image_newline': 'cuda:0',
            'model.vision_model': 'cuda:0',
            'model.language_model': 'cuda:0',
            'vision_encoder': 'cuda:0',
            'language_model': 'cuda:0',
            'multi_modal_projector': 'cuda:0',
            'multi_modal_projector.linear_1': 'cuda:0',
            'multi_modal_projector.linear_2': 'cuda:0',
            'multi_modal_projector.layer_norm': 'cuda:0',
            # Add vision tower components
            'vision_tower': 'cuda:0',
            'vision_tower.vision_model': 'cuda:0',
            'vision_tower.vision_model.embeddings': 'cuda:0',
            'vision_tower.vision_model.embeddings.class_embedding': 'cuda:0',
            'vision_tower.vision_model.embeddings.position_embedding': 'cuda:0',
            'vision_tower.vision_model.embeddings.patch_embedding': 'cuda:0',
            'vision_tower.vision_model.pre_layrnorm': 'cuda:0',
            'vision_tower.vision_model.encoder': 'cuda:0',
            'vision_tower.vision_model.post_layernorm': 'cuda:0',
            'vision_tower.mm_projector': 'cuda:0'
        }

        logger.info("Loading model with device map...")
        logger.info(f"Device map configuration: {device_map}")

        # Load model with more explicit configuration
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_config.name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={
                0: "12GB",
                "cpu": "48GB"
            },
            offload_folder="offload"  # Add offload folder for large models
        )

        # Enable gradient checkpointing
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Verify model loaded successfully
        logger.info("Successfully loaded model")
        logger.info(f"Model device map: {model.hf_device_map}")

        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        
        # Add more detailed error information
        if "device set" in str(e):
            logger.error("Device mapping issue detected. Component not properly mapped:")
            component = str(e).split("'")[0].strip()
            logger.error(f"Missing component: {component}")
        return None
    
def setup_processor(model_config: ModelConfig):
    """Setup and configure processor."""
    try:
        processor = AutoProcessor.from_pretrained(
            model_config.name,
            trust_remote_code=True
        )
        
        # Configure processor
        processor.image_processor.size = {'height': model_config.image_size, 'width': model_config.image_size}
        processor.image_processor.patch_size = model_config.patch_size
        processor.tokenizer.padding_side = 'right'
        
        return processor
    except Exception as e:
        logger.error(f"Error setting up processor: {e}")
        logger.error(traceback.format_exc())
        return None

def test_model_loading():
    """Test model loading and configuration."""
    try:
        # Inspect components first
        if not inspect_model_components():
            logger.warning("Could not inspect model components")
            
        # Check device availability
        if not check_device_availability():
            logger.warning("Testing will proceed with CPU only")
            
        model_config = ModelConfig()
        model = load_model_for_testing(model_config)
        
        if model is None:
            return False
            
        # Verify device mapping
        if not verify_model_devices(model):
            return False
            
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        return False

def test_processor_setup():
    """Test processor setup and configuration."""
    try:
        model_config = ModelConfig()
        processor = setup_processor(model_config)
        
        if processor is None:
            return False
            
        logger.info("Processor configured successfully")
        return True
    except Exception as e:
        logger.error(f"Processor setup test failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation and processing."""
    try:
        model_config = ModelConfig()
        processor = setup_processor(model_config)
        
        if processor is None:
            return False
            
        # Create small test dataset
        dataset = MemoryEfficientPlantDataset(
            processor=processor,
            split="train",
            sample_fraction=0.01,
            cache_dir="test_output/cache"
        )
        
        if len(dataset) == 0:
            logger.error("Dataset is empty")
            return False
            
        logger.info(f"Created dataset with {len(dataset)} samples")
        return True
    except Exception as e:
        logger.error(f"Dataset creation test failed: {e}")
        return False

def test_single_batch():
    """Test processing a single batch."""
    try:
        model_config = ModelConfig()
        processor = setup_processor(model_config)
        model = load_model_for_testing(model_config)
        
        if None in (processor, model):
            return False
            
        # Create test dataset
        dataset = MemoryEfficientPlantDataset(
            processor=processor,
            split="train",
            sample_fraction=0.01,
            cache_dir="test_output/cache"
        )
        
        # Get single batch
        batch = dataset[0]
        
        # Setup trainer
        training_args = get_training_args(
            "test_output",
            model_config,
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
        logger.info(f"Batch processed with loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Single batch test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_multiple_batches(num_batches=2):
    """Test processing multiple batches."""
    try:
        logger.info(f"Testing {num_batches} batches...")
        
        for i in range(num_batches):
            if not test_single_batch():
                logger.error(f"Batch {i+1} failed")
                return False
                
            # Cleanup between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"Batch {i+1} completed successfully")
            
        return True
    except Exception as e:
        logger.error(f"Multiple batch test failed: {e}")
        return False

def run_all_tests():
    """Run all tests in sequence."""
    try:
        # Setup test environment
        test_dir = setup_test_environment()
        
        # Log system information
        log_system_info()
        
        tests = [
            ("Model Loading", test_model_loading),
            ("Processor Setup", test_processor_setup),
            ("Dataset Creation", test_dataset_creation),
            ("Single Batch", test_single_batch),
            ("Multiple Batches", lambda: test_multiple_batches(2))
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\nRunning {test_name} test...")
            try:
                success = test_func()
                results.append((test_name, success))
                logger.info(f"{test_name} test: {'Passed' if success else 'Failed'}")
            except Exception as e:
                logger.error(f"{test_name} test failed with error: {e}")
                results.append((test_name, False))
            
            # Cleanup after each test
            cleanup_test_environment(test_dir)
        
        # Log final results
        logger.info("\nTest Results:")
        for test_name, success in results:
            logger.info(f"{test_name}: {'Passed' if success else 'Failed'}")
        
        return all(success for _, success in results)
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        logger.error(traceback.format_exc())
        return False
    
def verify_model_devices(model):
    """Verify all model components are properly mapped to devices."""
    try:
        device_issues = []
        
        # Check each named parameter
        for name, param in model.named_parameters():
            if param.device == torch.device("meta"):
                device_issues.append(f"{name} is on meta device")
            if param.device.type == "cpu" and "lm_head" not in name:
                logger.warning(f"{name} is on CPU")

        if device_issues:
            logger.error("Device mapping issues found:")
            for issue in device_issues:
                logger.error(f"  - {issue}")
            return False
            
        logger.info("All model components properly mapped to devices")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying model devices: {e}")
        return False
    
def check_device_availability():
    """Check and log available devices and memory."""
    logger.info("\nChecking device availability...")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            logger.info(f"CUDA Device {i}: {prop.name}")
            logger.info(f"  Total memory: {prop.total_memory / 1024**3:.2f}GB")
            logger.info(f"  Free memory: {torch.cuda.memory_allocated(i) / 1024**3:.2f}GB used")
            
        # Set default device
        torch.cuda.set_device(0)
        logger.info(f"Default CUDA device set to: cuda:0")
        return True
    else:
        logger.warning("No CUDA device available, falling back to CPU")
        return False

def inspect_model_components():
    """Inspect model components and their parameters."""
    try:
        config = AutoConfig.from_pretrained(ModelConfig().name)
        logger.info("Model architecture components:")
        for attr in dir(config):
            if not attr.startswith('_'):
                logger.info(f"  - {attr}")
        return True
    except Exception as e:
        logger.error(f"Error inspecting model components: {e}")
        return False

if __name__ == "__main__":
    # Check for HUGGINGFACE_TOKEN
    if not os.getenv("HUGGINGFACE_TOKEN"):
        logger.error("HUGGINGFACE_TOKEN environment variable not set")
        exit(1)
        
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    success = run_all_tests()
    exit(0 if success else 1)
# old_test_trainer.py
import os
import torch
import logging
import gc
import traceback
from huggingface_hub import login
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)
from models.trainer import CustomTrainer
from data.dataset import MemoryEfficientPlantDataset
from config.old_training_config import get_training_args, ModelConfig
from utils.tokenizer_utils import smart_tokenizer_and_embedding_resize

# Setup detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # Configure model padding
    model.config.padding_side = 'right'
    model.padding_side = 'right'
    
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

        # Load processor first
        logger.info("Loading and configuring processor...")
        processor = AutoProcessor.from_pretrained(
            model_config.name,
            trust_remote_code=True
        )
        
        # Explicitly configure processor
        processor.patch_size = 14
        processor.vision_feature_select_strategy = 'default'
        if hasattr(processor, 'image_processor'):
            processor.image_processor.patch_size = 14
            processor.image_processor.vision_feature_select_strategy = 'default'
            processor.image_processor.size = {'height': 336, 'width': 336}
            
        # Configure tokenizer
        processor.tokenizer.padding_side = 'right'
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        # Load model
        logger.info("Loading model...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_config.name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Configure model
        model.config.use_cache = False  # Disable KV cache for training
        model.config.padding_side = 'right'
        
        # Enable gradient checkpointing
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
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
        
        # Setup trainer with fp16 disabled
        logger.info("Setting up trainer...")
        training_args = get_training_args("test_output")
        training_args.fp16 = False  # Disable fp16 training
        training_args.bf16 = True   # Use bfloat16 instead
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=test_dataset
        )
        
        # Process batch
        logger.info("Processing batch through trainer...")
        loss = trainer.training_step(model, batch)
        logger.info(f"Successfully processed batch with loss: {loss}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    test_single_batch()
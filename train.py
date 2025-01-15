# train.py
import os
import gc
import torch
import traceback
import argparse
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration
)
import logging
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback


# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Import local modules
from data.dataset import MemoryEfficientPlantDataset
from models.trainer import CustomTrainer
from config.training_config import (
    get_training_args,
    ModelConfig,
    OptimizationConfig,
    DataConfig
)
from utils.logging_utils import setup_logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LLaVA model for plant identification')
    
    # Training parameters
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                       help='Fraction of dataset to use (0.0-1.0)')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Maximum number of training steps')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=16,
                       help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, 
                       default="llava-hf/llava-v1.6-mistral-7b-hf",
                       help='Model name or path')
    parser.add_argument('--image_size', type=int, default=336,
                       help='Image size for processing')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default="plant_models",
                       help='Output directory for models and logs')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Hardware parameters
    parser.add_argument('--gpu_memory_limit', type=str, default="12GiB",
                       help='GPU memory limit')
    
    return parser.parse_args()


def setup_model_and_processor(model_config: ModelConfig, args):
    """Setup model and processor with proper configuration."""
    try:
        logger = logging.getLogger(__name__)
        
        # Load processor first
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_config.name,
            trust_remote_code=True
        )
        
        # Configure processor with required attributes
        processor.patch_size = model_config.patch_size
        processor.vision_feature_select_strategy = 'default'
        
        # Set the attributes on the image processor as well
        if hasattr(processor, 'image_processor'):
            processor.image_processor.patch_size = model_config.patch_size
            processor.image_processor.vision_feature_select_strategy = 'default'
            processor.image_processor.size = {
                'height': model_config.image_size,
                'width': model_config.image_size
            }
        
        # Set attributes in the config
        if not hasattr(processor, 'config'):
            processor.config = type('ProcessorConfig', (), {})()
        processor.config.patch_size = model_config.patch_size
        processor.config.vision_feature_select_strategy = 'default'
        processor.config.image_size = model_config.image_size
        
        # Configure tokenizer
        processor.tokenizer.padding_side = 'right'  # For training
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        # Log processor configuration
        logger.info("Processor configuration:")
        logger.info(f"  patch_size: {getattr(processor, 'patch_size', None)}")
        logger.info(f"  vision_feature_select_strategy: {getattr(processor, 'vision_feature_select_strategy', None)}")
        logger.info(f"  image_processor.patch_size: {getattr(processor.image_processor, 'patch_size', None)}")
        logger.info(f"  config.patch_size: {getattr(processor.config, 'patch_size', None)}")
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load and configure model
        logger.info("Loading model...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_config.name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Configure model for training
        model.config.use_cache = False  # Required for gradient checkpointing
        model.config.padding_side = 'right'  # For training
        model.padding_side = 'right'
        
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Ensure model knows about processor configuration
        model.config.vision_config.patch_size = model_config.patch_size
        model.config.vision_config.image_size = model_config.image_size
        model.config.vision_feature_select_strategy = 'default'
        
        # Log model configuration
        logger.info("Model configuration:")
        logger.info(f"  padding_side: {model.padding_side}")
        logger.info(f"  use_cache: {model.config.use_cache}")
        logger.info(f"  patch_size: {model.config.vision_config.patch_size}")
        logger.info(f"  image_size: {model.config.vision_config.image_size}")
        
        return model, processor
        
    except Exception as e:
        logger.error(f"Error in setup_model_and_processor: {e}")
        logger.error(traceback.format_exc())
        raise

def prepare_datasets(processor, data_config: DataConfig, image_size: int, args):
    """Prepare training and evaluation datasets."""
    try:
        logger = logging.getLogger(__name__)
        
        logger.info("Preparing training dataset...")
        train_dataset = MemoryEfficientPlantDataset(
            processor=processor,
            split="train",
            sample_fraction=args.sample_fraction,
            image_size=image_size,
            cache_dir=os.path.join(args.output_dir, "cache", "train")
        )
        
        logger.info("Preparing evaluation dataset...")
        eval_dataset = MemoryEfficientPlantDataset(
            processor=processor,
            split="test",
            sample_fraction=args.sample_fraction,
            image_size=image_size,
            cache_dir=os.path.join(args.output_dir, "cache", "eval")
        )
        
        # Calculate class weights
        class_weights = train_dataset.get_class_weights()
        
        # Adjust class_weights to match total number of classes
        total_classes = 32064
        if class_weights.shape[0] != total_classes:
            logger.warning(
                f"Adjusting class_weights from shape {class_weights.shape} to match {total_classes} classes"
            )
            class_weights = torch.nn.functional.pad(
                class_weights, (0, total_classes - class_weights.shape[0]), value=1.0
            )
        
        logger.info(f"Final class weights shape: {class_weights.shape}")
        
        return train_dataset, eval_dataset, class_weights
        
    except Exception as e:
        logger.error(f"Error in prepare_datasets: {e}")
        logger.error(traceback.format_exc())
        raise

def validate_configurations(model, processor):
    """Validate model and processor configurations match."""
    logger = logging.getLogger(__name__)
    
    # Check processor configuration
    assert hasattr(processor, 'patch_size'), "Processor missing patch_size"
    assert hasattr(processor, 'vision_feature_select_strategy'), "Processor missing vision_feature_select_strategy"
    assert processor.tokenizer.padding_side == 'right', "Incorrect tokenizer padding side"
    
    # Check model configuration
    assert model.config.padding_side == 'right', "Incorrect model padding side"
    assert not model.config.use_cache, "use_cache should be False for training"
    
    logger.info("Configuration validation passed")

def prepare_for_inference(model, processor):
    """Prepare model and processor for inference by setting left padding."""
    model.config.padding_side = 'left'
    model.padding_side = 'left'
    processor.tokenizer.padding_side = 'left'
    model.config.use_cache = True  # Enable cache for inference
    return model, processor

def train_llava_model(args):
    """Main training function with enhanced error handling and dataset validation"""
    try:
        # Setup output directories
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(output_dir / "logs")
        logger = logging.getLogger(__name__)
        
        # Log training configuration
        logger.info("Training Configuration:")
        for arg, value in vars(args).items():
            logger.info(f"  {arg}: {value}")
        
        # Load configurations
        model_config = ModelConfig(name=args.model_name)
        optim_config = OptimizationConfig(
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.grad_accum_steps
        )
        data_config = DataConfig(
            train_batch_size=args.batch_size
        )
        
        if args.max_steps:
            optim_config.max_steps = args.max_steps
        
        # Aggressive memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Setup model and processor
        model, processor = setup_model_and_processor(model_config, args)
        
        # Validate configurations
        validate_configurations(model, processor)
        
        # Prepare datasets with validated processor
        train_dataset, eval_dataset, class_weights = prepare_datasets(
            processor,
            data_config,
            model_config.image_size,
            args
        )
        
        # Initialize trainer
        training_args = get_training_args(
            output_dir,
            model_config,
            optim_config,
            data_config
        )
        
        # Create trainer with validated model and processor
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            class_weights=class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        logger.info("Beginning model training...")
        trainer.train(resume_from_checkpoint=args.resume_from)
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save final model
        final_output_dir = output_dir / "final"
        final_output_dir.mkdir(exist_ok=True)
        
        logger.info("Saving final model...")
        trainer.save_model(final_output_dir)
        processor.save_pretrained(final_output_dir)
        
        logger.info(f"Training completed. Model saved to {final_output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    args = parse_arguments()
    train_llava_model(args)
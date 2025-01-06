"""
LLaVA 1.6 Fine-tuning Diagnostic Script
---------------------------------------
Debugging and analyzing data preparation for LLaVA model training
"""

import os
import torch
import numpy as np
from transformers import (
    CLIPImageProcessor,
    LlavaNextForConditionalGeneration,
    AutoTokenizer
)
from datasets import load_dataset
import logging
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def diagnose_dataset_processing():
    """
    Comprehensive diagnostic function to analyze dataset processing
    """
    try:
        # Load dataset
        logger.info("Loading Oxford Flowers dataset...")
        dataset = load_dataset("nelorth/oxford-flowers", split="train", trust_remote_code=True)
        
        # Prepare components
        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Categories mapping
        categories = {
            i: name for i, name in enumerate(dataset.features['label'].names)
        }
        
        # Sample processing diagnostics
        logger.info(f"Total samples in dataset: {len(dataset)}")
        
        # Test processing for multiple samples
        failure_count = 0
        success_count = 0
        
        for idx in range(min(10, len(dataset))):  # Test first 10 samples
            try:
                # Get item from dataset
                item = dataset[idx]
                
                # Image conversion
                if isinstance(item['image'], np.ndarray):
                    image = Image.fromarray(item['image'])
                elif isinstance(item['image'], Image.Image):
                    image = item['image']
                else:
                    image = Image.fromarray(np.array(item['image']))
                
                # Resize and convert
                image = image.convert('RGB')
                image = image.resize((224, 224))
                
                # Get label
                label_idx = item['label']
                label = categories[label_idx]
                
                # Prepare text
                prompt = f"You are a botanical expert. Please identify this plant. This is a {label}. It displays the characteristic features of this flower species."
                
                # Tokenize
                text_inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
                
                # Process image
                pixel_values = image_processor(
                    images=image, 
                    return_tensors="pt"
                ).pixel_values
                
                # Log successful processing
                logger.info(f"Successfully processed sample {idx}")
                success_count += 1
                
                # Optional: Log detailed sample information
                logger.debug(f"Sample {idx} details:")
                logger.debug(f"Image shape: {pixel_values.shape}")
                logger.debug(f"Input IDs shape: {text_inputs.input_ids.shape}")
                
            except Exception as sample_error:
                logger.error(f"Failed to process sample {idx}: {sample_error}")
                failure_count += 1
        
        # Summary
        logger.info(f"Processing Summary:")
        logger.info(f"Successful samples: {success_count}")
        logger.info(f"Failed samples: {failure_count}")
        
        if failure_count > 0:
            raise ValueError(f"Failed to process {failure_count} samples")
        
    except Exception as e:
        logger.error(f"Diagnostic process failed: {e}")
        import traceback
        traceback.print_exc()

def inspect_model_requirements():
    """
    Inspect LLaVA model input requirements
    """
    try:
        # Load model
        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Log model configuration details
        logger.info("Model Configuration Inspection:")
        logger.info(f"Model Config: {model.config}")
        
        # Attempt to access specific model attributes
        logger.info("Checking specific model attributes:")
        
        # List attributes related to image processing
        image_related_attrs = [
            'image_token_index', 
            'image_seq_length', 
            'num_image_tokens',
            'image_processor'
        ]
        
        for attr in image_related_attrs:
            try:
                value = getattr(model.config, attr, None)
                logger.info(f"{attr}: {value}")
            except Exception as attr_error:
                logger.warning(f"Could not access {attr}: {attr_error}")
        
    except Exception as e:
        logger.error(f"Model inspection failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main diagnostic entry point
    """
    logger.info("Starting LLaVA Model Diagnostic Process")
    
    # Run diagnostics
    diagnose_dataset_processing()
    inspect_model_requirements()
    
    logger.info("Diagnostic process completed")

if __name__ == "__main__":
    main()
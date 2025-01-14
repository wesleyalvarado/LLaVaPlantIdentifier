# utils/model_utils.py
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def configure_model_and_processor(model, processor) -> Tuple[object, object]:
    """Configure model and processor with correct settings.
    
    Args:
        model: The LLaVA-Next model
        processor: The model's processor
        
    Returns:
        Tuple of (configured_model, configured_processor)
    """
    # Configure processor
    if hasattr(model.config, 'vision_config'):
        patch_size = model.config.vision_config.image_size
        processor.patch_size = patch_size
        logger.info(f"Set processor patch_size to {patch_size}")
    
    processor.vision_feature_select_strategy = 'default'
    logger.info("Set processor vision_feature_select_strategy to 'default'")
    
    # Configure padding sides
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = 'right'
        logger.info("Set processor tokenizer padding_side to 'right'")
        
    if hasattr(model, 'padding_side'):
        model.padding_side = 'right'
        logger.info("Set model padding_side to 'right'")
    
    # Set special tokens if needed
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
        
    # Log current configuration
    logger.info("\nProcessor configuration:")
    logger.info(f"  patch_size: {getattr(processor, 'patch_size', 'Not set')}")
    logger.info(f"  vision_feature_select_strategy: {getattr(processor, 'vision_feature_select_strategy', 'Not set')}")
    logger.info(f"  tokenizer padding_side: {getattr(processor.tokenizer, 'padding_side', 'Not set')}")
    logger.info(f"  pad_token: {getattr(processor.tokenizer, 'pad_token', 'Not set')}")
    
    return model, processor
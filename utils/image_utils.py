# utils/image_utils.py
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

def select_best_resolution(current_size: Tuple[int, int], grid_pinpoints: List[List[int]]) -> Tuple[int, int]:
    """
    Always returns 336x336 as that's what LLaVA expects.
    This function is kept for API compatibility but no longer selects from grid_pinpoints.
    """
    logger.info("Using fixed 336x336 resolution for LLaVA")
    return (336, 336)

def image_size_to_num_patches(image_size: Tuple[int, int], patch_size: int) -> int:
    """
    Calculate the number of patches for a 336x336 image with 14x14 patches.
    This should always result in 576 patches (24x24) plus 1 CLS token.
    """
    if image_size != (336, 336):
        logger.warning(f"Expected 336x336 image, got {image_size}. Forcing 336x336.")
    
    # 336 / 14 = 24 patches in each dimension
    patches_per_side = 336 // patch_size  # Should be 24
    num_patches = (patches_per_side * patches_per_side) + 1  # Should be 577
    logger.info(f"Using {num_patches} patches (24x24 grid + 1 CLS token)")
    return num_patches


def prepare_image_inputs(
    pixel_values: torch.Tensor,
    model: nn.Module,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare image inputs for the LLaVA model.
    
    Args:
        pixel_values: Input tensor [C, H, W] or [B, C, H, W]
        model: The model
        device: Device to put tensors on
    
    Returns:
        Tuple of (pixel_values, image_sizes)
    """
    # Add batch dimension if needed
    if len(pixel_values.shape) == 3:
        pixel_values = pixel_values.unsqueeze(0)
    
    logger.info(f"Processing pixel_values shape: {pixel_values.shape}")
    
    # Create image sizes tensor [B, 2]
    batch_size = pixel_values.shape[0]
    image_sizes = torch.tensor([[336, 336]] * batch_size, device=device)
    
    logger.info(f"Final shapes:")
    logger.info(f"  pixel_values: {pixel_values.shape}")
    logger.info(f"  image_sizes: {image_sizes.shape}")
    
    return pixel_values, image_sizes

def validate_image_tensors(
    pixel_values: torch.Tensor,
    image_sizes: torch.Tensor
) -> bool:
    """Validate image tensors have correct shapes and types."""
    try:
        if len(pixel_values.shape) != 5:
            logger.error(f"pixel_values should be 5D [B, num_patches, C, H, W], got shape {pixel_values.shape}")
            return False
            
        if len(image_sizes.shape) != 2:
            logger.error(f"image_sizes should be 2D [B, 2], got shape {image_sizes.shape}")
            return False
            
        if image_sizes.shape[1] != 2:
            logger.error(f"image_sizes should have shape (batch_size, 2), got {image_sizes.shape}")
            return False
            
        # Check patch dimension is reasonable
        max_patches = 1000  # Set a reasonable maximum
        if pixel_values.shape[1] > max_patches:
            logger.error(f"Number of patches ({pixel_values.shape[1]}) exceeds maximum ({max_patches})")
            return False
        
        # Log tensor properties for debugging
        logger.info("Validation passed:")
        logger.info(f"  pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}")
        logger.info(f"  image_sizes: shape={image_sizes.shape}, dtype={image_sizes.dtype}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during tensor validation: {e}")
        return False
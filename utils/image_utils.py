# utils/image_utils.py
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

def select_best_resolution(current_size: Tuple[int, int], grid_pinpoints: List[List[int]]) -> Tuple[int, int]:
    """Select the best matching resolution from grid_pinpoints."""
    current_area = current_size[0] * current_size[1]
    best_resolution = sorted(
        grid_pinpoints, 
        key=lambda x: abs(x[0]*x[1] - current_area)
    )[0]
    logger.info(f"Selected resolution: {best_resolution}")
    return tuple(best_resolution)

def image_size_to_num_patches(image_size: Tuple[int, int], grid_pinpoints: List[List[int]], patch_size: int) -> int:
    """
    Calculate the number of patches after preprocessing for images of any resolution.
    Using the exact same calculation as the LLaVA-Next model.
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")
    
    height, width = image_size
    num_patches = 0
    # This matches the model's calculation exactly
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # Add the base patch
    num_patches += 1
    logger.info(f"Calculated {num_patches} patches for size {image_size} with patch size {patch_size}")
    return num_patches

def prepare_image_inputs(
    pixel_values: torch.Tensor,
    model: nn.Module,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare image inputs for the LLaVA-Next model."""
    # Add batch dimension if needed
    if len(pixel_values.shape) == 3:  # [C, H, W]
        pixel_values = pixel_values.unsqueeze(0)  # [1, C, H, W]
    
    # Get dimensions
    batch_size, channels, height, width = pixel_values.shape
    logger.info(f"Processing image of size {height}x{width}")
    
    # Get model configuration
    grid_pinpoints = model.config.image_grid_pinpoints
    
    # CLIP's patch size is 14
    patch_size = 14  # Fixed CLIP patch size
    
    # Select target resolution
    target_resolution = select_best_resolution(
        (height, width),
        grid_pinpoints
    )
    target_height, target_width = target_resolution
    
    # Calculate patches - each patch is 14x14, +1 for cls token
    h_patches = target_height // patch_size
    w_patches = target_width // patch_size
    num_patches = h_patches * w_patches + 1  # Add 1 for cls token
    
    logger.info(f"Target resolution: {target_resolution}")
    logger.info(f"Patch calculation: {h_patches}x{w_patches} + 1 = {num_patches} patches")
    
    # Create 5D tensor [batch_size, num_patches, channels, height, width]
    processed_pixels = pixel_values.unsqueeze(1).repeat(1, num_patches, 1, 1, 1)
    
    # Create image sizes tensor [batch_size, 2]
    image_sizes = torch.tensor([target_resolution], device=device)
    
    logger.info(f"Final pixel_values shape: {processed_pixels.shape}")
    logger.info(f"Final image_sizes shape: {image_sizes.shape}")
    
    return processed_pixels, image_sizes

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
            
        # Log tensor properties for debugging
        logger.info("Validation passed:")
        logger.info(f"  pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}")
        logger.info(f"  image_sizes: shape={image_sizes.shape}, dtype={image_sizes.dtype}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during tensor validation: {e}")
        return False
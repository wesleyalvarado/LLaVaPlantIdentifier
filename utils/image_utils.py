# utils/image_utils.py
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

def select_best_resolution(current_size: Tuple[int, int], grid_pinpoints: List[List[int]]) -> Tuple[int, int]:
    """Select the best matching resolution from grid_pinpoints.
    
    Args:
        current_size: Tuple of (height, width) for current image
        grid_pinpoints: List of [height, width] options from model config
        
    Returns:
        Tuple of (height, width) that best matches from grid_pinpoints
    """
    current_area = current_size[0] * current_size[1]
    best_resolution = sorted(
        grid_pinpoints, 
        key=lambda x: abs(x[0]*x[1] - current_area)
    )[0]
    return tuple(best_resolution)

def get_num_patches(image_size: Tuple[int, int], patch_size: int) -> int:
    """Calculate number of patches for given image size.
    
    Args:
        image_size: Tuple of (height, width)
        patch_size: Size of each patch
        
    Returns:
        Number of patches
    """
    height, width = image_size
    num_patches = 0
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # Add the base patch
    num_patches += 1
    return num_patches

def prepare_image_inputs(
    pixel_values: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    target_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare image inputs for the LLaVA-Next model.
    
    Args:
        pixel_values: Input image tensor
        model: The LLaVA-Next model (for configuration)
        device: Device to place tensors on
        target_size: Optional target size for resizing
        
    Returns:
        Tuple of (processed_pixel_values, image_sizes)
    """
    # Add batch dimension if needed
    if len(pixel_values.shape) == 3:  # [C, H, W]
        pixel_values = pixel_values.unsqueeze(0)  # [1, C, H, W]
    
    # Get dimensions
    batch_size, channels, height, width = pixel_values.shape
    logger.info(f"Working with image of size {height}x{width}")
    
    # Get the model's configuration
    grid_pinpoints = model.config.image_grid_pinpoints
    vision_config = model.config.vision_config
    patch_size = vision_config.image_size
    
    # Find best matching resolution
    best_resolution = select_best_resolution(
        (height, width),
        grid_pinpoints
    )
    logger.info(f"Selected resolution from grid_pinpoints: {best_resolution}")
    
    # Calculate number of patches
    num_patches = get_num_patches(best_resolution, patch_size)
    logger.info(f"Number of patches: {num_patches}")
    
    # Create 5D tensor with shape [batch_size, num_patches, channels, height, width]
    pixel_values = pixel_values.unsqueeze(1).repeat(1, num_patches, 1, 1, 1)
    image_sizes = torch.tensor([best_resolution], device=device)
    
    logger.info(f"Prepared pixel_values shape: {pixel_values.shape}")
    logger.info(f"Prepared image_sizes: {image_sizes.tolist()}")
    
    return pixel_values, image_sizes

def validate_image_tensors(
    pixel_values: torch.Tensor,
    image_sizes: torch.Tensor
) -> bool:
    """Validate image tensors have correct shapes and types.
    
    Args:
        pixel_values: The pixel values tensor
        image_sizes: The image sizes tensor
        
    Returns:
        bool: Whether tensors are valid
    """
    if len(pixel_values.shape) != 5:
        logger.error(f"pixel_values should be 5D, got shape {pixel_values.shape}")
        return False
        
    if len(image_sizes.shape) != 2:
        logger.error(f"image_sizes should be 2D, got shape {image_sizes.shape}")
        return False
        
    if image_sizes.shape[1] != 2:
        logger.error(f"image_sizes should have shape (batch_size, 2), got {image_sizes.shape}")
        return False
        
    return True
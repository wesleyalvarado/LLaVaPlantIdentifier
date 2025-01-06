import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

def check_tensor_shape(tensor: torch.Tensor, name: str = "") -> bool:
    """
    Debug helper to check tensor shape and contents
    
    Args:
        tensor: Input tensor
        name: Optional name for logging
    
    Returns:
        bool: Whether tensor is valid
    """
    if tensor is None:
        logger.debug(f"{name} is None")
        return False
        
    logger.debug(f"{name} shape: {tensor.shape}")
    logger.debug(f"{name} dtype: {tensor.dtype}")
    
    try:
        logger.debug(f"{name} min/max values: {tensor.min():.4f}/{tensor.max():.4f}")
    except:
        logger.debug(f"{name} min/max values could not be computed")
    
    return True

def validate_tensor_outputs(inputs: dict, expected_shapes: dict = None) -> bool:
    """
    Validate tensor shapes and check for NaN or infinite values
    
    Args:
        inputs: Dictionary of tensors to validate
        expected_shapes: Optional dictionary of expected shapes
    
    Returns:
        bool: Whether all tensors are valid
    """
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            logger.warning(f"{key} is not a tensor")
            continue
            
        # Check for NaN or infinite values
        if torch.isnan(value).any():
            logger.error(f"NaN values found in {key}")
            return False
        
        if torch.isinf(value).any():
            logger.error(f"Infinite values found in {key}")
            return False
        
        # Shape validation if expected shapes are provided
        if expected_shapes and key in expected_shapes:
            expected = expected_shapes[key]
            
            # Remove extra dimensions
            while value.ndim > len(expected):
                value = value.squeeze(0)
            
            # Check shape
            if value.shape[-len(expected):] != expected:
                logger.error(f"Incorrect shape for {key}: got {value.shape}, expected {expected}")
                return False
        
    return True

def fix_pixel_values_shape(pixel_values: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Fix common issues with pixel values tensor shape
    
    Args:
        pixel_values: Input pixel values tensor
        target_size: Desired image size
    
    Returns:
        Corrected pixel values tensor
    """
    try:
        # Extensive logging for debugging
        logger.debug("Entering fix_pixel_values_shape")
        logger.debug(f"Input pixel_values type: {type(pixel_values)}")
        logger.debug(f"Input pixel_values shape: {pixel_values.shape}")
        
        # Ensure we have a valid tensor
        if pixel_values is None:
            logger.error("Pixel values are None")
            return None
        
        # Convert to tensor if not already
        if not isinstance(pixel_values, torch.Tensor):
            try:
                pixel_values = torch.tensor(pixel_values)
            except Exception as conv_error:
                logger.error(f"Failed to convert to tensor: {conv_error}")
                return None
        
        # Specific handling for [3, 3, 336, 336] shape
        if pixel_values.shape == (3, 3, target_size, target_size):
            logger.debug("Detected [3, 3, 336, 336] shape, reshaping")
            # Take the first of the 3 channels
            pixel_values = pixel_values[0]
        
        # Remove extra dimensions
        while pixel_values.ndim > 4:
            pixel_values = pixel_values.squeeze(0)
        
        # Ensure 4D tensor: [batch, channels, height, width]
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        
        # Validate final shape
        if pixel_values.ndim != 4:
            logger.error(f"Unable to standardize pixel values shape. Current shape: {pixel_values.shape}")
            return None
        
        # Additional shape checks
        if pixel_values.shape[1] != 3:  # Ensure 3 color channels
            logger.error(f"Unexpected number of channels: {pixel_values.shape[1]}")
            return None
        
        # Resize if needed
        if pixel_values.shape[2] != target_size or pixel_values.shape[3] != target_size:
            logger.debug(f"Resizing pixel values from {pixel_values.shape} to (3, {target_size}, {target_size})")
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )
        
        return pixel_values
    
    except Exception as e:
        logger.error(f"Error fixing pixel values shape: {e}")
        return None

def validate_processed_sample(processed, target_size: int):
    """
    Validate and standardize processed sample
    
    Args:
        processed: Processed sample 
        target_size: Expected image size
    
    Returns:
        Validated and corrected sample or None
    """
    try:
        # Extensive logging for debugging
        logger.debug("Entering validate_processed_sample")
        logger.debug(f"Input type: {type(processed)}")
        
        # Convert to dictionary if not already
        if not isinstance(processed, dict):
            try:
                logger.debug("Attempting to convert to dictionary")
                processed = dict(processed)
            except Exception as conv_error:
                logger.warning(f"Failed to convert to dictionary: {conv_error}")
                return None
        
        # Log available keys
        logger.debug(f"Available keys: {list(processed.keys())}")
        
        # Validate key components
        required_keys = ['pixel_values', 'input_ids', 'attention_mask']
        for key in required_keys:
            if key not in processed:
                logger.warning(f"Missing required key: {key}")
                return None
        
        # Validate pixel values
        pixel_values = processed['pixel_values']
        fixed_pixel_values = fix_pixel_values_shape(pixel_values, target_size)
        if fixed_pixel_values is None:
            logger.warning("Failed to fix pixel values shape")
            return None
        
        # Ensure correct shapes for all tensors
        for key in ['input_ids', 'attention_mask']:
            tensor = processed[key]
            # Squeeze extra dimensions if present
            while tensor.ndim > 2:
                tensor = tensor.squeeze(0)
            processed[key] = tensor
        
        # Update processed dict with fixed pixel values
        processed['pixel_values'] = fixed_pixel_values
        
        return processed
    
    except Exception as e:
        logger.error(f"Comprehensive sample validation error: {e}")
        return None

def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor values to 0-1 range
    
    Args:
        tensor: Input tensor
    
    Returns:
        Normalized tensor
    """
    try:
        # Handle different input types
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        
        # Normalize to 0-1 range
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Avoid division by zero
        if min_val == max_val:
            return torch.zeros_like(tensor, dtype=torch.float32)
        
        normalized = (tensor - min_val) / (max_val - min_val)
        return normalized.float()
    
    except Exception as e:
        logger.error(f"Tensor normalization error: {e}")
        return None
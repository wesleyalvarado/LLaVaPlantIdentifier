import numpy as np
from PIL import Image
import logging
import torch

logger = logging.getLogger(__name__)

def validate_image_data(image_data: np.ndarray, idx: int) -> bool:
    """
    Validate numpy image data with comprehensive checks
    
    Args:
        image_data: Input numpy array
        idx: Index of the image for logging
    
    Returns:
        bool: Whether image data is valid
    """
    try:
        # Basic None check
        if image_data is None:
            logger.error(f"No image data for index {idx}")
            return False
        
        # Ensure numpy array
        if not isinstance(image_data, np.ndarray):
            try:
                image_data = np.array(image_data)
            except Exception as conv_error:
                logger.error(f"Failed to convert to numpy array at index {idx}: {conv_error}")
                return False
        
        # Log original image details
        logger.debug(f"Original image shape for idx {idx}: {image_data.shape}")
        logger.debug(f"Image dtype: {image_data.dtype}")
        
        # Dimensionality check
        if image_data.ndim not in [2, 3]:
            logger.error(f"Invalid image dimensions at index {idx}. Got shape: {image_data.shape}")
            return False
        
        # Size check
        if image_data.size == 0:
            logger.error(f"Empty image data at index {idx}")
            return False
        
        # NaN check
        if np.any(np.isnan(image_data)):
            logger.error(f"NaN values detected in image for idx {idx}")
            return False
        
        # Channel check for 3D arrays
        if image_data.ndim == 3 and image_data.shape[2] not in [1, 3, 4]:
            logger.error(f"Unexpected number of channels at index {idx}. Got: {image_data.shape[2]}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Unexpected error validating image at index {idx}: {e}")
        return False

def convert_to_pil_image(image_data: np.ndarray, idx: int) -> Image.Image:
    """
    Convert numpy array to PIL Image with robust handling
    
    Args:
        image_data: Input image data
        idx: Index of the image for logging
    
    Returns:
        PIL Image or None
    """
    try:
        # First validate the image data
        if not validate_image_data(image_data, idx):
            logger.error(f"Image validation failed for index {idx}")
            return None
        
        # Ensure numpy array
        if not isinstance(image_data, np.ndarray):
            image_data = np.array(image_data)
        
        # Normalize floating point images
        if image_data.dtype in [np.float32, np.float64]:
            image_data = ((image_data - image_data.min()) / 
                          (image_data.max() - image_data.min()) * 255).astype(np.uint8)
        
        # Handle different dimensional cases
        if image_data.ndim == 2:
            # Grayscale to RGB
            image_data = np.stack([image_data]*3, axis=-1)
        elif image_data.ndim == 3 and image_data.shape[2] > 3:
            # Truncate to 3 channels if more exist
            image_data = image_data[:, :, :3]
        
        # Convert to PIL Image
        image = Image.fromarray(image_data.astype(np.uint8), mode='RGB')
        
        logger.debug(f"Converted image at index {idx}:")
        logger.debug(f"  Mode: {image.mode}")
        logger.debug(f"  Size: {image.size}")
        
        return image
    
    except Exception as e:
        logger.error(f"Failed to convert image at index {idx}: {e}")
        return None

def process_pil_image(image: Image.Image, target_size: int, idx: int = None) -> Image.Image:
    """
    Process PIL image to correct mode and size
    
    Args:
        image: Input PIL Image
        target_size: Desired image size
        idx: Optional index for logging
    
    Returns:
        Processed PIL Image or None
    """
    try:
        # Validate input is a PIL Image
        if not isinstance(image, Image.Image):
            logger.error(f"Invalid image type{' at index ' + str(idx) if idx is not None else ''}. Expected PIL Image.")
            return None
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            logger.debug(f"Converting image from {image.mode} to RGB{' at index ' + str(idx) if idx is not None else ''}")
            image = image.convert('RGB')
        
        # Resize image
        resized_image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Validate resizing
        if resized_image.size != (target_size, target_size):
            logger.error(f"Resizing failed{' at index ' + str(idx) if idx is not None else ''}. Got {resized_image.size}")
            return None
        
        logger.debug(f"Processed image{' at index ' + str(idx) if idx is not None else ''}:")
        logger.debug(f"  Mode: {resized_image.mode}")
        logger.debug(f"  Size: {resized_image.size}")
        
        return resized_image
    
    except Exception as e:
        logger.error(f"Image processing error{' at index ' + str(idx) if idx is not None else ''}: {e}")
        return None
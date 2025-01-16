# utils/model_optimizer.py
import torch
import logging
from typing import Optional, Dict
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Utilities for optimizing LLaVA model training."""
    
    @staticmethod
    def optimize_memory_usage(model: PreTrainedModel, max_memory: Optional[Dict[int, str]] = None) -> PreTrainedModel:
        """Optimize model memory usage with advanced techniques."""
        try:
            # Enable memory efficient attention
            if hasattr(model, "enable_memory_efficient_attention"):
                model.enable_memory_efficient_attention()
                logger.info("Enabled memory efficient attention")
            
            # Enable model parallel if multiple GPUs available
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                logger.info(f"Enabled DataParallel across {torch.cuda.device_count()} GPUs")
            
            # Enable gradient checkpointing
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
            
            # Optimize CUDA memory allocation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Reserve some memory for system
                
            return model
            
        except Exception as e:
            logger.error(f"Error optimizing memory usage: {e}")
            return model
    
    @staticmethod
    def optimize_training_settings(model: PreTrainedModel) -> PreTrainedModel:
        """Apply optimal training settings."""
        try:
            # Freeze vision tower for initial training
            if hasattr(model, "vision_tower"):
                for param in model.vision_tower.parameters():
                    param.requires_grad = False
                logger.info("Froze vision tower parameters")
            
            # Use adaptive loss scaling
            model.config.use_cache = False  # Disable KV-cache during training
            
            # Enable model fusion optimizations if available
            if hasattr(model, "enable_fused_layernorm"):
                model.enable_fused_layernorm()
                logger.info("Enabled fused LayerNorm")
            
            return model
            
        except Exception as e:
            logger.error(f"Error optimizing training settings: {e}")
            return model
    
    @staticmethod
    def setup_mixed_precision(model: PreTrainedModel) -> PreTrainedModel:
        """Configure mixed precision training."""
        try:
            # Use bfloat16 if available, otherwise fall back to float16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model = model.to(torch.bfloat16)
                logger.info("Enabled bfloat16 mixed precision")
            else:
                model = model.to(torch.float16)
                logger.info("Enabled float16 mixed precision")
                
            return model
            
        except Exception as e:
            logger.error(f"Error setting up mixed precision: {e}")
            return model

    @staticmethod
    def optimize_batch_processing(model: PreTrainedModel) -> PreTrainedModel:
        """Optimize batch processing settings."""
        try:
            # Enable tensor cores if available
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TensorFloat-32 inference")
            
            # Set optimal forward pass settings
            model.config.use_return_dict = False  # Slight memory optimization
            
            return model
            
        except Exception as e:
            logger.error(f"Error optimizing batch processing: {e}")
            return model
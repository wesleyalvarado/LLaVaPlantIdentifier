import os
import gc
import torch
import numpy as np
import traceback
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig
)
from datasets import load_dataset
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Union, Any
import torch.nn as nn
import torch.nn.functional as F

# Aggressive memory management
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# At the top level of your file, add this debug function
def check_tensor_shape(tensor, name=""):
    """Debug helper to check tensor shape and contents"""
    if tensor is None:
        logger.debug(f"{name} is None")
        return False
        
    logger.debug(f"{name} shape: {tensor.shape}")
    logger.debug(f"{name} min/max values: {tensor.min():.4f}/{tensor.max():.4f}")
    return True


class MemoryEfficientPlantDataset(Dataset):
    def __init__(self, processor, split="train", sample_fraction=0.1, image_size=336):
        logger.info(f"Loading {split} dataset...")
        full_dataset = load_dataset("nelorth/oxford-flowers", split=split, trust_remote_code=True)
        
        num_samples = max(10, int(len(full_dataset) * sample_fraction))
        self.dataset = full_dataset.select(range(num_samples))
        
        self.processor = processor
        self.image_size = image_size
        logger.debug(f"Using image size: {self.image_size}")
        
        self.categories = {
            i: name for i, name in enumerate(self.dataset.features['label'].names)
        }
        
        logger.info(f"Loaded {len(self.dataset)} images for {split}")
        logger.info(f"Found {len(self.categories)} plant categories")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            logger.debug(f"Processing item {idx}")
            
            # Debug original image data
            if isinstance(item['image'], np.ndarray):
                logger.debug(f"Original image shape for idx {idx}: {item['image'].shape}")
                logger.debug(f"Image dtype: {item['image'].dtype}")
                if np.any(np.isnan(item['image'])):
                    logger.error(f"NaN values detected in image for idx {idx}")
                    return None
            
            # Ensure we have valid image data
            if item['image'] is None:
                logger.error(f"No image data for index {idx}")
                return None
                
            # Convert image data to PIL Image and validate
            try:
                if isinstance(item['image'], np.ndarray):
                    image = Image.fromarray(item['image'])
                else:
                    image = Image.fromarray(np.array(item['image']))
                logger.debug(f"PIL Image mode: {image.mode}, size: {image.size}")
            except Exception as img_error:
                logger.error(f"Image conversion failed for index {idx}: {img_error}")
                return None
            
            # Validate and convert image mode
            if image.mode != 'RGB':
                logger.debug(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Resize image to match model's expected size
            if image.size != (self.image_size, self.image_size):
                logger.debug(f"Resizing image from {image.size} to ({self.image_size}, {self.image_size})")
                image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            
            # Get label information
            label_idx = item['label']
            label = self.categories[label_idx]
            logger.debug(f"Label index: {label_idx}, category: {label}")
            
            prompt = f"Identify this {label} flower."
            
            # Process through processor with detailed debugging
           try:
    logger.debug(f"Running processor for idx {idx}")
    
    # Process image first with step by step debugging
    image_features = self.processor.image_processor(
        image,
        do_resize=False,  # We handle resizing
        do_center_crop=False,
        do_rescale=True,
        do_normalize=True,
        return_tensors="pt"
    )
    
    # Debug the raw output
    pixel_values = image_features['pixel_values']
    logger.debug(f"Raw processor output shape: {pixel_values.shape}")
    
    # Handle extra dimensions if needed
    if len(pixel_values.shape) == 5:  # [1, 3, 3, H, W]
        pixel_values = pixel_values.squeeze(0)  # Remove batch dim
        pixel_values = pixel_values[0]  # Take first of repeating channels
        logger.debug(f"After shape correction: {pixel_values.shape}")
    
    # Final validation
    check_tensor_shape(pixel_values, "Final pixel_values")
    
    if pixel_values.shape != (3, self.image_size, self.image_size):
        logger.error(f"Shape mismatch: got {pixel_values.shape}, expected (3, {self.image_size}, {self.image_size})")
        return None
                
                # Validate pixel values shape
                if inputs['pixel_values'].shape != (3, self.image_size, self.image_size):
                    logger.error(f"Incorrect pixel values shape: {inputs['pixel_values'].shape}")
                    return None
                
                # Ensure all required keys are present
                required_keys = {'input_ids', 'attention_mask', 'pixel_values', 'labels'}
                if not all(k in inputs for k in required_keys):
                    logger.error(f"Missing required keys: {required_keys - set(inputs.keys())}")
                    return None
                
                return inputs
                
            except Exception as proc_error:
                logger.error(f"Processor failed for index {idx}: {proc_error}")
                logger.error(f"Processor error traceback: {traceback.format_exc()}")
                return None
                
        except Exception as e:
            logger.error(f"Sample processing failed for index {idx}: {e}")
            return None

def memory_efficient_collate_fn(batch):
    # Remove None items and validate batch with debugging
    batch = [item for item in batch if item is not None]
    
    if not batch:
        logger.warning("No valid items in the batch, returning None")
        return None
    
    try:
        # Debug batch contents
        logger.debug(f"Collating batch of size {len(batch)}")
        logger.debug(f"First item keys: {batch[0].keys()}")
        
        # Get all required keys from the first item
        required_keys = {'input_ids', 'attention_mask', 'pixel_values', 'labels'}
        item_keys = set(batch[0].keys())
        
        if not required_keys.issubset(item_keys):
            missing_keys = required_keys - item_keys
            logger.error(f"Missing required keys in batch item: {missing_keys}")
            return None
        
        # Debug tensor properties before stacking
        for i, item in enumerate(batch):
            logger.debug(f"Item {i} shapes:")
            for key in required_keys:
                if key in item:
                    logger.debug(f"  {key}: shape={item[key].shape}, dtype={item[key].dtype}, device={item[key].device}")
                    # Check for NaN values
                    if torch.isnan(item[key]).any():
                        logger.error(f"NaN values found in {key} for item {i}")
                        return None
                else:
                    logger.error(f"Missing key {key} in item {i}")
                    return None
            
        # Stack tensors with validation
        batched = {}
        for key in required_keys:
            try:
                stacked = torch.stack([item[key] for item in batch])
                logger.debug(f"Successfully stacked {key}: shape={stacked.shape}")
                batched[key] = stacked
            except Exception as e:
                logger.error(f"Failed to stack {key}: {str(e)}")
                logger.error(f"Shapes for {key}: {[item[key].shape for item in batch]}")
                return None
        
        # Final validation of batched tensors
        logger.debug("Final batch shapes:")
        for key, tensor in batched.items():
            logger.debug(f"  {key}: {tensor.shape}")
            if torch.isnan(tensor).any():
                logger.error(f"NaN values found in batched {key}")
                return None
        
        return batched
    except Exception as e:
        logger.error(f"Batch creation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    
    except Exception as e:
        logger.error(f"Batch creation failed: {e}")
        return None

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=memory_efficient_collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], *args) -> torch.Tensor:
        """
        Perform a training step with comprehensive debugging.
        """
        model.train()
        
        # Debug inputs before preparation
        logger.debug("Training step input debug:")
        if inputs is None:
            logger.warning("Inputs is None")
            return torch.tensor(0.0, device=self.args.device)
        
        logger.debug(f"Input keys: {inputs.keys()}")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f"{key} shape: {value.shape}, dtype: {value.dtype}, device: {value.device}")
                if torch.isnan(value).any():
                    logger.error(f"NaN values found in {key}")
                    return torch.tensor(0.0, device=self.args.device)
        
        # Prepare inputs and debug again
        try:
            inputs = self._prepare_inputs(inputs)
            logger.debug("After preparation:")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    logger.debug(f"{key} shape: {value.shape}, dtype: {value.dtype}, device: {value.device}")
            
            # Validate prepared inputs
            if 'pixel_values' not in inputs or inputs['pixel_values'] is None:
                logger.error("Missing or None pixel_values after preparation")
                return torch.tensor(0.0, device=self.args.device)
            
            # Compute loss with additional debugging
            with self.compute_loss_context_manager():
                try:
                    logger.debug("Computing model outputs...")
                    loss = self.compute_loss(model, inputs)
                    logger.debug(f"Loss value: {loss.item() if loss is not None else 'None'}")
                except Exception as e:
                    logger.error(f"Error in loss computation: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise

            if self.args.n_gpu > 1:
                loss = loss.mean()

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                loss = loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.deepspeed:
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()

            return loss.detach()

        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def train(self):
        try:
            return super().train()
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Last successful batch: {self.state.log_history[-1] if self.state.log_history else 'No successful batches'}")
            raise

def train_llava_model():
    """Main training function with extreme memory efficiency"""
    try:
        # Aggressive memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        # Model and output directories
        model_dir = os.path.expanduser("~/plant_models/llava_plant_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Model configuration
        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        logger.info(f"Loading model and processor: {model_name}")
        
        # Load processor first
        processor = LlavaNextProcessor.from_pretrained(model_name)
        logger.debug(f"Image processor type: {type(processor.image_processor).__name__}")
        logger.debug(f"Text tokenizer type: {type(processor.tokenizer).__name__}")
        
        # Load model configuration first
        model_config = AutoConfig.from_pretrained(model_name)  # Changed from transformers.AutoConfig
        logger.debug(f"Vision config: {model_config.vision_config}")
        logger.debug(f"Text config: {model_config.text_config}")
        
        # Store vision config for dataset
        vision_config = model_config.vision_config
        image_size = vision_config.image_size
        logger.info(f"Using image size from config: {image_size}")
        
        # Load model with trust_remote_code=True
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            config=model_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code=True  # Important for loading custom model classes
        )
        
        # Verify critical model components
        model_has_vision = (hasattr(model, 'vision_model') or 
                          hasattr(model, 'vision_tower') or 
                          hasattr(model, 'vision_projection'))
        if not model_has_vision:
            logger.error("Vision components not found in model")
            logger.debug(f"Available model attributes: {dir(model)}")
            raise ValueError("Vision model components missing from LLaVA model")
            
        if not hasattr(model, 'language_model'):
            logger.error("Language model not properly loaded")
            raise ValueError("Language model missing from LLaVA model")
        
        # Freeze base parameters
        logger.info("Freezing base parameters...")
        for param in model.parameters():
            param.requires_grad = False
        
        # Selectively unfreeze language model parameters
        if hasattr(model, 'language_model'):
            logger.info("Unfreezing language model parameters...")
            for param in model.language_model.parameters():
                param.requires_grad = True
                
        # Log parameter status
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Prepare datasets with correct image size
        train_dataset = MemoryEfficientPlantDataset(
            processor=processor, 
            split="train", 
            sample_fraction=0.1,
            image_size=image_size
        )
        eval_dataset = MemoryEfficientPlantDataset(
            processor=processor, 
            split="test", 
            sample_fraction=0.1,
            image_size=image_size
        )
        
        # Training arguments with extreme memory optimization
        training_args = TrainingArguments(
            output_dir=model_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1024,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir=os.path.join(model_dir, "logs"),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            push_to_hub=False,
            fp16=False,
            bf16=False,
            optim="adamw_torch",
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            dataloader_num_workers=0
        )
        
        # Initialize CustomTrainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=memory_efficient_collate_fn,
        )
        
        # Start training
        logger.info("Beginning model training...")
        trainer.train()
        
        # Save final model
        final_output_dir = os.path.join(model_dir, "final")
        trainer.save_model(final_output_dir)
        processor.save_pretrained(final_output_dir)
        
        logger.info(f"Training completed. Model saved to {final_output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Aggressive cleanup
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train_llava_model()
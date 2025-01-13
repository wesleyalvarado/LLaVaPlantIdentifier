# test_trainer.py
import os
import torch
import logging
import psutil
import gc
import traceback
from huggingface_hub import login
from transformers import (
    AutoProcessor, 
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)
from models.trainer import CustomTrainer
from data.dataset import MemoryEfficientPlantDataset
from config.training_config import get_training_args, ModelConfig

# Setup detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_memory_stats():
    """Print current memory usage statistics"""
    logger.info("\nMemory Statistics:")
    # CPU Memory
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024**3
    logger.info(f"CPU Memory Usage: {cpu_memory:.2f} GB")
    
    # GPU Memory
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
        logger.info(f"GPU Memory Cached: {gpu_memory_cached:.2f} GB")

def clear_memory():
    """Clear unused memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def test_single_batch():
    """Test processing a single batch through the trainer"""
    try:
        # Print initial memory stats
        print_memory_stats()
        
        # First authenticate with Hugging Face
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")
        login(token)
        
        try:
            # Aggressive memory clearing before model loading
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                # Force garbage collection
                gc.collect()
                
                # Set memory allocation strategy
                torch.cuda.set_per_process_memory_fraction(0.5)  # Use only 50% of available GPU memory
                
                # Check available GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                logger.info(f"Total GPU memory: {gpu_memory / 1024**3:.2f} GB")
                
            # Set environment variables for memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
            
            logger.info("Loading processor...")
            processor = AutoProcessor.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf",
                trust_remote_code=True
            )
            
            logger.info("Loading model...")
            model_config = ModelConfig()
            
            # Configure 4-bit quantization with CPU offload
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_threshold=6.0,
                bnb_4bit_compute_type=torch.float16
            )

            # Configure device map for manual layer placement
            device_map = {
                'transformer.word_embeddings': 'cpu',
                'transformer.word_embeddings_layernorm': 'cpu',
                'transformer.final_layernorm': 'cpu',
                'transformer.prefix_encoder': 'cpu',
                'lm_head': 'cpu'
            }
            
            # Load model with aggressive memory optimizations
            logger.info("Starting model load with quantization...")
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_config.name,
                quantization_config=quantization_config,
                device_map="auto",  # Let it automatically handle remaining layers
                torch_dtype=torch.float16,
                trust_remote_code=True,
                offload_folder="offload",
                offload_state_dict=True,  # Enable state dict offloading
                low_cpu_mem_usage=True
            )
            
            # Enable gradient checkpointing and set to eval mode initially
            model.gradient_checkpointing_enable()
            model.eval()  # Start in eval mode to save memory
            
            # Print memory stats after model load
            print_memory_stats()
            
            logger.info("Creating test dataset...")
            test_dataset = MemoryEfficientPlantDataset(
                split="train",
                sample_fraction=0.01,  # Use minimal data for testing
                processor=processor
            )
            
            # Get a single batch
            logger.info("Getting single batch...")
            batch = test_dataset[0]
            
            # Print tensor shapes
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"{key} shape: {value.shape}")
                else:
                    logger.info(f"{key}: {value}")
            
            # Setup trainer
            logger.info("Setting up trainer...")
            training_args = get_training_args("test_output")
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=test_dataset
            )
            
            # Clear memory before processing batch
            clear_memory()
            
            # Process single batch
            logger.info("Processing batch through trainer...")
            try:
                # Set to train mode only when needed
                model.train()
                loss = trainer.training_step(model, batch)
                logger.info(f"Successfully processed batch with loss: {loss}")
                
                # Set back to eval mode
                model.eval()
                print_memory_stats()
                
            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                logger.error(traceback.format_exc())
                raise
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            raise
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    test_single_batch()
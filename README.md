# LLaVA Plant Training Application Documentation

## Project Overview
This application fine-tunes a LLaVA-NeXT (Large Language and Vision Assistant) model on the oxford_flowers102 dataset for plant identification. The project uses a memory-efficient approach to handle image processing, model training, and dataset management.

## Key Features
- Integrated LLaVA-NeXT model support
- Memory-efficient data processing
- Robust error handling and logging
- Automatic tensor shape management
- Comprehensive image validation
- 4-bit quantization support

## Project Structure
```
llava_plants/
├── train.py                 # Main training script
├── test_trainer.py         # Test script for validation
├── data/
│   ├── __init__.py
│   └── dataset.py          # Advanced dataset handling
├── models/
│   ├── __init__.py
│   └── trainer.py          # Custom trainer implementation
├── utils/
│   ├── __init__.py
│   ├── image_utils.py      # Image processing utilities
│   ├── tensor_utils.py     # Tensor validation
│   ├── tokenizer_utils.py  # Tokenizer configuration
│   └── logging_utils.py    # Logging setup
└── config/
    ├── __init__.py
    └── training_config.py  # Training configuration
```

## Prerequisites

### Hardware Requirements
- Minimum 50GB RAM
- GPU with at least 16GB VRAM (required)
- Storage: 50GB free space

### Software Requirements
- Python 3.10+
- CUDA 11.7+ (for GPU support)
- Git LFS (for model checkpoints)

## Setup Instructions

1. Create and activate conda environment:
```bash
conda create -n plant_vision python=3.10
conda activate plant_vision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export HUGGINGFACE_TOKEN="your_token_here"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false
```

## Pre-training Validation

Before running full training, execute the test script:
```bash
python test_trainer.py
```

Verify the following in the output:
- Image token processing is correct (look for "Number of image tokens: 1")
- Tensor shapes are consistent (especially pixel_values)
- Loss value is reasonable (typically between 0.5-2.0)
- No memory errors or leaks

## Training Configuration

### Current Optimal Settings
- Batch Size: 1
- Gradient Accumulation Steps: 16
- Learning Rate: 1e-5
- Mixed Precision: fp16
- Training Strategy: Gradient checkpointing enabled
- Memory Optimization: 4-bit quantization

### Resource Management
- GPU Memory: 90% allocation for model, 10% for buffer
- Gradient Clipping: 0.5
- Memory-efficient dataset loading
- Automatic garbage collection

## Common Issues and Solutions

### Memory Management
- If encountering OOM errors, increase gradient accumulation steps
- Enable 4-bit quantization using BitsAndBytesConfig
- Monitor GPU memory usage with `nvidia-smi`

### Image Processing
- Ensure images are correctly resized to 336x336
- Validate patch size is consistently 14x14
- Check for correct number of patches (576 + 1 CLS token)

### Training Stability
- Monitor loss values for sudden spikes
- Verify gradient norms remain under 1.0
- Check attention masks for correct padding

## Monitoring and Debugging

### Key Metrics to Watch
- Training Loss: Should decrease steadily
- GPU Memory Usage: Should remain stable
- Image Token Processing: Verify correct token counts
- Gradient Norms: Should not explode

### Logging
- All major operations are logged at INFO level
- Critical errors include full stack traces
- Performance metrics are logged every 10 steps
- Tensor shapes are validated at each stage

## Performance Optimization Tips

1. Data Loading
   - Use appropriate sample_fraction for dataset size
   - Enable pin_memory for GPU training
   - Adjust num_workers based on CPU cores

2. Model Configuration
   - Enable gradient checkpointing
   - Use 4-bit quantization
   - Implement proper garbage collection
   - Monitor memory usage

3. Training Process
   - Start with small sample_fraction
   - Gradually increase batch size if stable
   - Monitor loss curves for convergence
   - Save checkpoints regularly

## Future Improvements

1. Technical Enhancements
   - Implement dynamic batch sizing
   - Add multi-GPU support
   - Optimize memory management
   - Enhance error recovery

2. Features
   - Add validation metrics
   - Implement early stopping
   - Add model export utilities
   - Enhance logging visualization

## Contribution Guidelines

1. Code Style
   - Follow PEP 8 guidelines
   - Add comprehensive docstrings
   - Include type hints
   - Maintain extensive logging

2. Testing
   - Add unit tests for new features
   - Run test_trainer.py before commits
   - Validate memory usage
   - Check tensor shapes

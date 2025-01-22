# LLaVA Plant Training Application

## Project Overview
This application fine-tunes a LLaVA-NeXT (Large Language and Vision Assistant) model on the oxford_flowers102 dataset for plant identification. The project emphasizes robust error handling, comprehensive validation, and memory-efficient processing.

## Project Structure
```
llava_plants/
├── train.py                 # Main training script
├── test_trainer.py          # Test script for validation
├── data/
│   ├── __init__.py
│   └── dataset.py          # Memory-efficient dataset implementation
├── models/
│   ├── __init__.py
│   └── trainer.py          # Custom trainer implementation
├── utils/
│   ├── __init__.py
│   ├── dataset_validator.py # Dataset validation and statistics
│   ├── image_utils.py      # Image processing utilities
│   ├── tensor_utils.py     # Tensor validation and normalization
│   ├── tokenizer_utils.py  # Tokenizer configuration
│   ├── model_optimizer.py  # Model optimization utilities
│   └── logging_utils.py    # Logging setup
└── config/
    ├── __init__.py
    ├── base_config.py      # Base configuration classes
    ├── model_config.py     # Model-specific settings
    ├── training_config.py  # Training hyperparameters
    └── data_config.py      # Dataset and processing settings
```

## Core Components

### Utilities
- **dataset_validator.py**: Pre-training dataset validation
  - Statistical analysis of dataset
  - Memory usage validation
  - Class distribution analysis 
  - Image quality checks

- **tensor_utils.py**: Tensor operations and validation
  - Shape validation
  - NaN/Inf checking
  - Tensor normalization
  - Memory-efficient tensor operations

- **model_optimizer.py**: Model optimization utilities
  - Memory usage optimization
  - Training settings optimization
  - Mixed precision setup
  - Batch processing optimization

- **logging_utils.py**: Structured logging system
  - Training metrics logging
  - Error tracking
  - Performance monitoring
  - Checkpoint logging

### Data Management
- **dataset.py**: Memory-efficient dataset implementation
  - Lazy loading
  - Caching support
  - Basic validation
  - Memory cleanup

### Model Training
- **trainer.py**: Custom training implementation
  - Gradient accumulation
  - Memory-efficient training
  - Comprehensive error handling
  - Training state management

## Setup Instructions

### Prerequisites
- Python 3.10+
- CUDA 11.7+ (for GPU support)
- Git LFS (for model checkpoints)
- 16GB+ RAM
- GPU with 8GB+ VRAM (recommended)

### Environment Setup
```bash
conda create -n plant_vision python=3.10
conda activate plant_vision
pip install -r requirements.txt
```

### Environment Variables
```bash
export HUGGINGFACE_TOKEN="your_token_here"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false
```

## Usage

### Dataset Validation
Run validation before training to ensure data quality:
```bash
python -m utils.dataset_validator --data_dir path/to/data
```

### Training
Start training with custom parameters:
```bash
python train.py \
  --sample_fraction 1.0 \
  --batch_size 1 \
  --grad_accum_steps 16 \
  --learning_rate 1e-5
```

### Testing
Run validation tests:
```bash
python test_trainer.py
```

## Best Practices

### Memory Management
- Use gradient checkpointing for large models
- Enable 4-bit quantization
- Monitor GPU memory usage
- Implement proper garbage collection

### Training Process
1. Validate dataset before training
2. Start with small sample_fraction
3. Monitor loss curves
4. Save checkpoints regularly

### Error Handling
- Comprehensive error logging
- Graceful failure recovery
- Memory cleanup on errors
- Validation at critical points

## Common Issues and Solutions

### Memory Issues
- Increase gradient accumulation steps
- Enable 4-bit quantization
- Monitor GPU memory with nvidia-smi
- Use memory-efficient dataset loading

### Training Stability
- Monitor loss values
- Verify gradient norms
- Check attention masks
- Validate tensor shapes

## Contributing
1. Follow PEP 8 guidelines
2. Add comprehensive docstrings
3. Include type hints
4. Maintain extensive logging
5. Run test_trainer.py before commits

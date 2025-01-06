# LLaVA Plant Training Project Documentation

## Project Overview
This project implements a training pipeline for fine-tuning a LLaVA (Large Language and Vision Assistant) model on the Oxford Flowers dataset for plant identification. The project is designed with memory efficiency as a primary consideration, particularly for systems with limited resources like M2 Macs.

## Project Structure
```
llava_plants/
├── train.py                 # Main training script
├── data/
│   ├── __init__.py
│   └── dataset.py          # Advanced dataset handling
├── models/
│   ├── __init__.py
│   └── trainer.py          # Custom trainer implementation
├── utils/
│   ├── __init__.py
│   ├── image_utils.py      # Image processing utilities
│   ├── tensor_utils.py     # Tensor validation and handling
│   └── logging_utils.py    # Logging configuration
└── config/
    ├── __init__.py
    └── training_config.py  # Training configuration
```

## Component Details

### 1. models/trainer.py

#### CustomTrainer
Extends HuggingFace's Trainer class with custom functionality for plant identification.

##### Key Methods:
- `get_train_dataloader()`
  - Creates DataLoader with memory-efficient settings
  - Implements custom batch creation
  - Manages memory during data loading

- `training_step()`
  - Handles individual training iterations
  - Processes model inputs
  - Computes loss and manages gradients
  - Includes comprehensive debugging
  - Implements error handling

#### memory_efficient_collate_fn()
- Combines individual samples into batches
- Validates tensor shapes
- Handles None values
- Efficiently stacks tensors for training

### 2. data/dataset.py

#### MemoryEfficientPlantDataset
Implements memory-efficient dataset handling for the Oxford Flowers dataset.

##### Key Methods:
- `__init__()`
  - Initializes dataset with processor
  - Loads Oxford Flowers dataset
  - Sets up category mappings
  - Configures memory settings

- `__getitem__()`
  - Retrieves and processes samples
  - Converts images to correct format
  - Creates identification prompts
  - Returns processed tensors

- `__len__()`
  - Returns dataset size

#### Dataset Collate Function
- Combines samples into batches
- Handles tensor stacking
- Implements memory cleanup
- Manages error cases

### 3. utils/tensor_utils.py

#### Key Functions:

##### check_tensor_shape()
- Validates tensor dimensions
- Logs tensor properties
- Verifies tensor integrity

##### validate_tensor_outputs()
- Checks for NaN/infinite values
- Validates against expected shapes
- Ensures tensor quality

##### fix_pixel_values_shape()
- Corrects image tensor shapes
- Standardizes dimensions
- Ensures proper model input format

##### validate_processed_sample()
- Validates complete samples
- Verifies required components
- Ensures proper formatting

##### normalize_tensor()
- Normalizes values to 0-1 range
- Handles various input types
- Includes error handling

### 4. config/training_config.py

#### ModelConfig
Configuration dataclass for model parameters:
- Model name
- Image size
- Data type
- Hardware mapping

#### get_training_args()
Creates training arguments:
- Learning rates
- Batch sizes
- Optimization settings
- Logging configuration
- Evaluation parameters

## Memory Management Features

### Key Strategies:
1. Streaming Data Loading
   - Loads data incrementally
   - Reduces memory footprint
   - Manages resource usage

2. Memory Cleanup
   - Regular garbage collection
   - Tensor cleanup
   - Cache clearing

3. Tensor Validation
   - Shape verification
   - Type checking
   - Error handling

4. Efficient Batch Processing
   - Optimized collation
   - Memory-aware batching
   - Resource monitoring

5. Error Handling
   - Comprehensive logging
   - Error recovery
   - Debug information

## Training Flow

1. Initialization
   - Load configuration
   - Setup logging
   - Initialize model

2. Data Processing
   - Load dataset
   - Process images
   - Create prompts

3. Training Loop
   - Batch creation
   - Forward pass
   - Loss computation
   - Optimization

4. Monitoring
   - Progress tracking
   - Resource usage
   - Error logging

## Memory Optimization Tips

1. Model Configuration
   - Use appropriate batch sizes
   - Enable gradient checkpointing
   - Optimize tensor precision

2. Data Handling
   - Implement streaming
   - Clean up unused tensors
   - Manage cache effectively

3. Resource Management
   - Monitor memory usage
   - Implement cleanup routines
   - Handle errors gracefully
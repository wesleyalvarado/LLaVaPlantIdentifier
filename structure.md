llava_plants/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── dataset.py         # MemoryEfficientPlantDataset
├── models/
│   ├── __init__.py
│   └── trainer.py         # CustomTrainer and collate function
├── utils/
│   ├── __init__.py
│   ├── image_utils.py     # Image processing helpers
│   ├── tensor_utils.py    # Tensor validation and debugging
│   └── logging_utils.py   # Logging configuration
├── config/
│   ├── __init__.py
│   └── training_config.py # Training arguments and model config
└── train.py              # Main training script
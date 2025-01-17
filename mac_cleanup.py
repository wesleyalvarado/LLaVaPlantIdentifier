# mac_cleanup.py
import os
import gc
import torch
import psutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def cleanup_mac_resources():
    """Clean up system resources after testing."""
    try:
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clear any cached files
        cache_dirs = [
            "test_output/cache",
            "test_output/checkpoints",
            ".pytest_cache",
            "__pycache__"
        ]
        
        for cache_dir in cache_dirs:
            path = Path(cache_dir)
            if path.exists():
                try:
                    for file in path.glob("**/*"):
                        if file.is_file():
                            file.unlink()
                    path.rmdir()
                    logger.info(f"Cleaned up {cache_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean {cache_dir}: {e}")
        
        # Log memory usage
        process = psutil.Process(os.getpid())
        logger.info(f"Memory usage after cleanup: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleanup_mac_resources()
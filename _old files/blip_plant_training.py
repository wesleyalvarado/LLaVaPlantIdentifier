import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import shutil
import json
from huggingface_hub import snapshot_download

class ModelDownloader:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.model_name = model_name
        self.save_path = "local_plant_model"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        
    def download_model(self):
        """Download model files using snapshot_download"""
        try:
            print("Downloading model files...")
            cache_dir = snapshot_download(
                repo_id=self.model_name,
                local_dir=self.save_path
            )
            return True
        except Exception as e:
            print(f"Error downloading model files: {e}")
            return False
            
    def load_and_verify(self):
        """Load and verify the model works"""
        try:
            print("\nLoading model to verify...")
            processor = BlipProcessor.from_pretrained(self.save_path)
            model = BlipForConditionalGeneration.from_pretrained(
                self.save_path,
                torch_dtype=torch.float32
            )
            
            # Save processor config explicitly
            processor_config = {
                "image_size": 224,
                "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711]
            }
            
            with open(os.path.join(self.save_path, "preprocessor_config.json"), "w") as f:
                json.dump(processor_config, f)
                
            return True
        except Exception as e:
            print(f"Error verifying model: {e}")
            return False

def main():
    # Clean up any existing files
    if os.path.exists("local_plant_model"):
        print("Removing existing model directory...")
        shutil.rmtree("local_plant_model")
    
    downloader = ModelDownloader()
    
    print("\nStep 1: Downloading model files...")
    if downloader.download_model():
        print("\nStep 2: Verifying model...")
        if downloader.load_and_verify():
            print("\nModel installation completed successfully!")
        else:
            print("\nModel verification failed.")
    else:
        print("\nModel download failed.")

if __name__ == "__main__":
    main()
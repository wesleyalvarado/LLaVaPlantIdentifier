# Import necessary libraries
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import requests
from io import BytesIO
import logging

# Suppress all logging
logging.getLogger('PIL').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)

# Set device (MPS for M2 Mac, otherwise CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load processor and model
processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

# Custom collate function to handle variable-sized tensors
def custom_collate(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    # If batch is empty after filtering
    if len(batch) == 0:
        return None
    
    # Find maximum lengths
    max_input_ids_len = max(item['input_ids'].shape[0] for item in batch)
    
    # Pad tensors
    padded_batch = {
        'input_ids': [],
        'attention_mask': [],
        'pixel_values': [],
    }
    
    for item in batch:
        # Pad input_ids
        input_ids = item['input_ids']
        pixel_values = item['pixel_values']
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Ensure consistent sizes
        padded_input_ids = torch.nn.functional.pad(
            input_ids, 
            (0, max_input_ids_len - input_ids.shape[0]), 
            value=processor.tokenizer.pad_token_id
        )
        padded_attention_mask = torch.nn.functional.pad(
            attention_mask, 
            (0, max_input_ids_len - attention_mask.shape[0]), 
            value=0
        )
        
        # Reshape pixel values to 4D tensor (batch, channels, height, width)
        padded_pixel_values = pixel_values.unsqueeze(0) if pixel_values.ndim == 3 else pixel_values
        
        padded_batch['input_ids'].append(padded_input_ids)
        padded_batch['attention_mask'].append(padded_attention_mask)
        padded_batch['pixel_values'].append(padded_pixel_values)
    
    # Convert to tensors
    padded_batch['input_ids'] = torch.stack(padded_batch['input_ids'])
    padded_batch['attention_mask'] = torch.stack(padded_batch['attention_mask'])
    padded_batch['pixel_values'] = torch.stack(padded_batch['pixel_values'])
    
    return padded_batch

# Custom Dataset wrapper for image-text data
class ImageTextDataset(Dataset):
    def __init__(self, dataset, processor, max_length=128):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        
        # Filter out problematic entries
        self.valid_indices = []
        for idx in range(len(dataset)):
            try:
                # Attempt to load image
                self._load_image(idx)
                self.valid_indices.append(idx)
            except Exception:
                continue
    
    def _load_image(self, idx):
        # Try to load image from URL
        item = self.dataset[idx]
        try:
            # First try to download image
            response = requests.get(item['image_url'], timeout=5)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception:
            raise ValueError("Could not load image")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get actual dataset index
        real_idx = self.valid_indices[idx]
        
        # Get image and caption
        item = self.dataset[real_idx]
        
        try:
            # Download image
            response = requests.get(item['image_url'], timeout=5)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Prepare inputs
            inputs = self.processor(
                images=image, 
                text=item['caption'], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                legacy=False  # Use new behavior
            )
            
            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "pixel_values": inputs["pixel_values"].squeeze(),
            }
        except Exception:
            # Silent failure for individual items
            return None

# Prepare dataset and dataloader
def create_dataloader(split='train', batch_size=4, num_samples=None):
    # Load dataset
    dataset = load_dataset("conceptual_captions", split=split)
    
    # Limit number of samples if specified
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Create custom dataset
    custom_dataset = ImageTextDataset(dataset, processor)
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        custom_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,  # Drop last incomplete batch
        collate_fn=custom_collate
    )
    
    return dataloader

# Training function
def train_model(model, dataloader, optimizer, device, num_epochs=3):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        for batch in dataloader:
            # Skip empty batches
            if batch is None:
                continue
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=input_ids
                )
                
                # Compute loss
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Optimize
                optimizer.step()
                
                # Track loss
                total_loss += loss.item()
                batch_count += 1
            
            except Exception:
                continue
        
        # Print epoch summary
        if batch_count > 0:
            print(f"Epoch {epoch+1}, Average Loss: {total_loss / batch_count:.4f}")

# Main training script
def main():
    # Create dataloader
    train_dataloader = create_dataloader(split='train', batch_size=4, num_samples=1000)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Train the model
    train_model(model, train_dataloader, optimizer, device)
    
    # Save the model
    model.save_pretrained("./image_text_model")
    processor.save_pretrained("./image_text_processor")
    print("Training completed and model saved!")

# Inference function
def generate_caption(image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Prepare inputs
    inputs = processor(image, return_tensors="pt", legacy=False)
    
    # Generate caption
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    
    # Decode caption
    generated_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_caption

# Run the training
if __name__ == "__main__":
    main()
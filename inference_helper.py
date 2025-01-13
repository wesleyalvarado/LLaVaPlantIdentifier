import torch
from PIL import Image
import requests
from io import BytesIO
import logging
from typing import Union, Optional
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlavaHelper:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def load_image(self, image_source: Union[str, Image.Image]) -> Optional[Image.Image]:
        """Load image from file, URL, or PIL Image"""
        try:
            if isinstance(image_source, Image.Image):
                return image_source
            elif image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source)
                return Image.open(BytesIO(response.content))
            else:
                return Image.open(image_source)
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def generate_response(self, 
                         image_source: Union[str, Image.Image], 
                         prompt: str,
                         max_length: int = 512,
                         temperature: float = 0.7) -> str:
        """Generate response for an image and prompt"""
        try:
            # Load and process image
            image = self.load_image(image_source)
            if image is None:
                return "Error: Could not load image"

            # Convert image to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Prepare inputs
            inputs = self.model.build_conversation_input_ids(
                self.tokenizer, 
                query=prompt,
                images=[image]
            )
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            # Generate response
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_length=max_length,
                use_cache=True
            )

            # Decode response
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response = response.split("ASSISTANT: ")[-1].strip()
            
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error generating response: {str(e)}"

    def analyze_image(self, image_source: Union[str, Image.Image]) -> str:
        """Analyze an image with a default prompt"""
        default_prompt = "Please describe this image in detail. What do you see?"
        return self.generate_response(image_source, default_prompt)

# Example usage:
def create_helper(model, tokenizer):
    """Create a LlavaHelper instance"""
    return LlavaHelper(model, tokenizer)
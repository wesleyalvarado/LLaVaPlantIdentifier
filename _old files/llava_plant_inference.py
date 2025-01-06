import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantAnalyzer:
    def __init__(self, model_path="./image_text_model", processor_path="./image_text_processor"):
        logger.info(f"Loading model from {model_path}...")
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.cuda.is_available() else 
                                 "cpu")
        
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        logger.info(f"Model loaded successfully on {self.device}")

    def analyze_plant(self, image_path: str, question: str = None) -> str:
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path).convert('RGB')
            
            # Enhanced default question with more specific prompting
            if not question:
                question = "What is the specific name or type of this flower? For example, is it a rose, daisy, tulip, or another variety? Please identify it and describe its appearance."
            else:
                # Enhance user's question to encourage more specific answers
                question = f"Looking at this flower image, {question} Please be specific about the flower variety if you can identify it."
            
            # Process image
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                legacy=False
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with adjusted parameters for more detailed output
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,     # Increased for more detailed responses
                    min_new_tokens=20,      # Ensure more substantial responses
                    num_beams=5,            # Increased for more thorough search
                    do_sample=True,
                    temperature=0.7,        # Slightly increased for more variety
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    length_penalty=1.2,     # Encourage slightly longer responses
                    early_stopping=True
                )
            
            # Decode and clean output
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output while preserving important plant information
            if '*' in generated_text:
                generated_text = generated_text.split('*')[0]
            if ':' in generated_text:
                # Only split on ':' if it's not part of plant description
                if not any(plant_type in generated_text.lower() for plant_type in ['daisy', 'rose', 'tulip']):
                    generated_text = generated_text.split(':')[0]
            
            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return f"Error analyzing image: {str(e)}"

def main():
    try:
        analyzer = PlantAnalyzer()
        
        print("\nAdvanced Plant Analysis System (type 'quit' to exit)")
        print("Using locally trained model")
        print("-" * 50)
        
        while True:
            image_path = input("\nEnter path to plant image or URL (or 'quit' to exit): ")
            
            if image_path.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            question = input("Enter your question about the plant (or press Enter for default): ").strip()
            
            print("\nAnalyzing image...")
            result = analyzer.analyze_plant(image_path, question if question else None)
            
            print("\nAnalysis Results:")
            print("-" * 50)
            print(result)
            print("-" * 50)

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
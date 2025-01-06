from PIL import Image
from transformers import pipeline
import torch

class PlantAnalyzer:
    def __init__(self):
        print("Loading pre-trained vision model...")
        try:
            # Use BLIP-2 model instead of LLaVA
            self.pipe = pipeline(
                "image-to-text",
                model="Salesforce/blip2-opt-2.7b",
                device_map="mps" if torch.backends.mps.is_available() else "cpu"
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def analyze_plant(self, image_path, query=None):
        """
        Analyze a plant image using BLIP-2
        
        Args:
            image_path: Path to image file
            query: Specific question about the plant (optional)
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Default prompt if none provided
            if not query:
                prompt = ("Describe this plant in detail. What species might it be? "
                         "Describe its characteristics, appearance, and any notable features.")
            else:
                prompt = query

            # Get model's analysis
            result = self.pipe(
                image, 
                prompt,
                max_new_tokens=100,
                top_k=50,
                temperature=0.7,
                num_beams=5
            )
            
            return result[0]['generated_text']
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

    def suggest_questions(self):
        """Return list of useful questions to ask about plants"""
        return [
            "What type of plant is this?",
            "What are the main features of this plant?",
            "Describe the leaves and flowers of this plant.",
            "What growing conditions does this plant prefer?",
            "What are the distinctive characteristics of this plant?",
            "Is this an indoor or outdoor plant?",
            "What season does this plant bloom in?"
        ]

def main():
    # Initialize the analyzer
    analyzer = PlantAnalyzer()
    
    print("\nPlant Analysis System (type 'quit' to exit)")
    print("-" * 50)
    
    # Show suggested questions
    print("\nSuggested questions you can ask:")
    for question in analyzer.suggest_questions():
        print(f"- {question}")
    
    while True:
        # Get image path
        image_path = input("\nEnter path to plant image (or 'quit' to exit): ")
        
        if image_path.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        # Get optional specific question
        question = input("Enter your question (or press Enter for default analysis): ").strip()
        
        print("\nAnalyzing image...")
        result = analyzer.analyze_plant(image_path, question if question else None)
        
        if result:
            print("\nAnalysis Results:")
            print("-" * 50)
            print(result)
            print("-" * 50)

if __name__ == "__main__":
    main()
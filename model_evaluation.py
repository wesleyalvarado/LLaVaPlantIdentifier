# evaluate.py
## Not tested yet
import torch
from PIL import Image
import os
from pathlib import Path
import random
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from data.dataset import MemoryEfficientPlantDataset

class ModelEvaluator:
    def __init__(self, model_path: str):
        """Initialize evaluator with trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # Load test dataset
        self.test_dataset = MemoryEfficientPlantDataset(
            processor=self.processor,
            split="test",
            sample_fraction=1.0  # Use full test set for evaluation
        )
        
    def evaluate_single_image(self, image_path: str) -> dict:
        """Evaluate a single image with confidence score."""
        image = Image.open(image_path)
        inputs = self.processor(
            images=image,
            text="<image>\nWhat type of flower is shown in this image? Please identify the flower species.",
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=100
            )
            
        # Get prediction and confidence
        prediction = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)
        confidence = torch.mean(outputs.scores[0]).item()
        
        return {
            "prediction": prediction,
            "confidence": confidence
        }
    
    def evaluate_test_set(self, num_samples: int = None) -> dict:
        """Evaluate model on test set."""
        results = []
        true_labels = []
        predictions = []
        confidences = []
        
        # Use subset of test set if specified
        test_indices = range(len(self.test_dataset))
        if num_samples:
            test_indices = random.sample(test_indices, min(num_samples, len(self.test_dataset)))
        
        for idx in tqdm(test_indices, desc="Evaluating"):
            sample = self.test_dataset[idx]
            true_label = sample['class_name']
            
            # Get model prediction
            inputs = {
                'pixel_values': sample['pixel_values'].unsqueeze(0).to(self.device),
                'input_ids': sample['input_ids'].unsqueeze(0).to(self.device),
                'attention_mask': sample['attention_mask'].unsqueeze(0).to(self.device)
            }
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=100
                )
            
            prediction = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)
            confidence = torch.mean(outputs.scores[0]).item()
            
            results.append({
                'true_label': true_label,
                'predicted': prediction,
                'confidence': confidence
            })
            
            true_labels.append(true_label)
            predictions.append(prediction)
            confidences.append(confidence)
        
        # Calculate metrics
        metrics = {
            'results': results,
            'classification_report': classification_report(true_labels, predictions),
            'confusion_matrix': confusion_matrix(true_labels, predictions),
            'average_confidence': sum(confidences) / len(confidences),
            'high_confidence_accuracy': self._calculate_high_confidence_accuracy(results)
        }
        
        return metrics
    
    def _calculate_high_confidence_accuracy(self, results, threshold=0.8):
        """Calculate accuracy for high confidence predictions."""
        high_conf_results = [r for r in results if r['confidence'] > threshold]
        if not high_conf_results:
            return 0.0
        
        correct = sum(1 for r in high_conf_results if r['true_label'] == r['predicted'])
        return correct / len(high_conf_results)
    
    def generate_evaluation_report(self, metrics: dict, output_dir: str):
        """Generate detailed evaluation report with visualizations."""
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save classification report
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(metrics['classification_report'])
        
        # Create confusion matrix plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()
        
        # Create confidence distribution plot
        confidences = [r['confidence'] for r in metrics['results']]
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50)
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.savefig(output_dir / 'confidence_distribution.png')
        plt.close()
        
        # Save detailed results to CSV
        df = pd.DataFrame(metrics['results'])
        df.to_csv(output_dir / 'detailed_results.csv', index=False)
        
        # Generate summary report
        summary = f"""
        Model Evaluation Summary
        -----------------------
        Total samples evaluated: {len(metrics['results'])}
        Average confidence: {metrics['average_confidence']:.3f}
        High confidence accuracy: {metrics['high_confidence_accuracy']:.3f}
        
        See classification_report.txt for detailed metrics.
        """
        
        with open(output_dir / 'summary.txt', 'w') as f:
            f.write(summary)

def main():
    # Example usage
    model_path = "path/to/your/saved/model"  # Replace with your model path
    output_dir = "evaluation_results"
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path)
    
    # Run evaluation
    print("Running full test set evaluation...")
    metrics = evaluator.evaluate_test_set()
    
    # Generate report
    print("Generating evaluation report...")
    evaluator.generate_evaluation_report(metrics, output_dir)
    
    # Example of single image evaluation
    image_path = "path/to/test/image.jpg"  # Replace with test image path
    result = evaluator.evaluate_single_image(image_path)
    print(f"\nSingle image test result:")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    main()
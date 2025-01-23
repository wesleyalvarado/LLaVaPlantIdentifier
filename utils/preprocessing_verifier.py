# utils/preprocessing_verifier.py
import torch
import logging
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class PreprocessingVerifier:
    """Verify and debug preprocessing pipeline for LLaVA training."""
    
    def __init__(self, processor):
        self.processor = processor
        
    def verify_image_processing(self, image_input) -> Dict[str, Any]:
        """Verify image preprocessing steps."""
        results = {}
        
        try:
            # Check input
            if isinstance(image_input, Image.Image):
                results['input_size'] = image_input.size
                results['input_mode'] = image_input.mode
            elif isinstance(image_input, np.ndarray):
                results['input_shape'] = image_input.shape
                results['input_dtype'] = str(image_input.dtype)
                results['input_range'] = (float(image_input.min()), float(image_input.max()))
            
            # Process image
            image_inputs = self.processor.image_processor(
                image_input,
                return_tensors="pt"
            )
            pixel_values = image_inputs['pixel_values']
            
            # Verify processed values
            results['processed_shape'] = tuple(pixel_values.shape)
            results['processed_dtype'] = str(pixel_values.dtype)
            results['value_range'] = (
                float(pixel_values.min()),
                float(pixel_values.max())
            )
            results['mean'] = float(pixel_values.mean())
            results['std'] = float(pixel_values.std())
            
            # Check for potential issues
            self._check_image_issues(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image verification: {e}")
            return {'error': str(e)}
    
    def _check_image_issues(self, results: Dict):
        """Check for common image processing issues."""
        issues = []
        
        # Value range check
        if results.get('value_range'):
            min_val, max_val = results['value_range']
            if min_val < -1.1 or max_val > 1.1:  # Allow slight buffer for numerical precision
                issues.append(f"Unusual value range: [{min_val:.2f}, {max_val:.2f}]")
        
        # Shape check
        if results.get('processed_shape'):
            if len(results['processed_shape']) != 4:  # [batch, channels, height, width]
                issues.append(f"Unexpected tensor shape: {results['processed_shape']}")
            elif results['processed_shape'][1] != 3:  # RGB channels
                issues.append(f"Unexpected number of channels: {results['processed_shape'][1]}")
        
        results['issues'] = issues
    
    def verify_text_processing(self, text_input: str) -> Dict[str, Any]:
        """Verify text preprocessing steps."""
        results = {}
        
        try:
            # Process text
            text_inputs = self.processor.tokenizer(
                text_input,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Basic token info
            results['input_text'] = text_input
            results['input_length'] = len(text_input)
            results['token_count'] = len(text_inputs['input_ids'][0])
            
            # Decode back to verify
            decoded = self.processor.tokenizer.decode(text_inputs['input_ids'][0])
            results['decoded_text'] = decoded
            
            # Special token verification
            special_tokens = {
                'pad_token': self.processor.tokenizer.pad_token,
                'eos_token': self.processor.tokenizer.eos_token,
                'bos_token': self.processor.tokenizer.bos_token,
                'unk_token': self.processor.tokenizer.unk_token
            }
            results['special_tokens'] = {k: v for k, v in special_tokens.items() if v is not None}
            
            # Check attention mask
            if 'attention_mask' in text_inputs:
                attention_mask = text_inputs['attention_mask'][0]
                results['attention_length'] = attention_mask.sum().item()
                results['padding_length'] = len(attention_mask) - attention_mask.sum().item()
            
            # Check for potential issues
            self._check_text_issues(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in text verification: {e}")
            return {'error': str(e)}
    
    def _check_text_issues(self, results: Dict):
        """Check for common text processing issues."""
        issues = []
        
        # Check for truncation
        if results.get('token_count', 0) >= self.processor.tokenizer.model_max_length:
            issues.append("Text was truncated to model max length")
        
        # Check for unknown tokens
        if results.get('decoded_text'):
            unk_token = self.processor.tokenizer.unk_token
            if unk_token and unk_token in results['decoded_text']:
                issues.append("Unknown tokens present in processed text")
        
        results['issues'] = issues
    
    def verify_end_to_end(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Verify complete preprocessing pipeline for a sample."""
        logger.info("Starting end-to-end preprocessing verification")
        
        verification = {}
        
        # Verify pixel values
        if 'pixel_values' in sample:
            logger.info("Verifying pixel values...")
            pixel_values = sample['pixel_values']
            if torch.is_tensor(pixel_values):
                verification['pixel_values'] = {
                    'shape': tuple(pixel_values.shape),
                    'dtype': str(pixel_values.dtype),
                    'range': (float(pixel_values.min()), float(pixel_values.max())),
                    'mean': float(pixel_values.mean()),
                    'std': float(pixel_values.std())
                }
        
        # Verify input IDs
        if 'input_ids' in sample:
            logger.info("Verifying input IDs...")
            input_ids = sample['input_ids']
            if torch.is_tensor(input_ids):
                decoded = self.processor.tokenizer.decode(input_ids[0] if input_ids.dim() > 1 else input_ids)
                verification['input_ids'] = {
                    'shape': tuple(input_ids.shape),
                    'decoded_text': decoded,
                    'sequence_length': len(input_ids[0] if input_ids.dim() > 1 else input_ids)
                }
        
        # Verify attention mask
        if 'attention_mask' in sample:
            logger.info("Verifying attention mask...")
            attention_mask = sample['attention_mask']
            if torch.is_tensor(attention_mask):
                verification['attention_mask'] = {
                    'shape': tuple(attention_mask.shape),
                    'active_tokens': int(attention_mask.sum().item()),
                    'padding_tokens': int((~attention_mask.bool()).sum().item())
                }
        
        # Verify labels
        if 'labels' in sample:
            logger.info("Verifying labels...")
            labels = sample['labels']
            if torch.is_tensor(labels):
                verification['labels'] = {
                    'shape': tuple(labels.shape),
                    'unique_values': sorted([int(x) for x in labels.unique().tolist()]),
                    'padding_tokens': int((labels == -100).sum().item())
                }
        
        # Check for issues
        verification['issues'] = self._check_end_to_end_issues(verification)
        
        return verification
    
    def _check_end_to_end_issues(self, verification: Dict) -> list:
        """Check for issues in the complete pipeline."""
        issues = []
        
        # Check pixel values
        if 'pixel_values' in verification:
            pv = verification['pixel_values']
            if pv['range'][0] < -1.1 or pv['range'][1] > 1.1:
                issues.append(f"Unusual pixel value range: {pv['range']}")
            
            # Check for expected shape with patches
            if len(pv['shape']) != 5:  # [batch, patches, channels, height, width]
                issues.append(f"Unexpected pixel values shape: {pv['shape']}")
        
        # Check sequence lengths
        if 'input_ids' in verification and 'attention_mask' in verification:
            input_shape = verification['input_ids']['shape']
            mask_shape = verification['attention_mask']['shape']
            if input_shape != mask_shape:
                issues.append(f"Mismatched shapes: input_ids {input_shape} vs attention_mask {mask_shape}")
        
        # Check label consistency
        if 'labels' in verification:
            labels = verification['labels']
            if -100 not in labels['unique_values']:
                issues.append("No padding token (-100) found in labels")
        
        return issues

def verify_preprocessing_pipeline(processor, sample_data):
    """Convenience function to run verification on sample data."""
    verifier = PreprocessingVerifier(processor)
    
    # Run verification
    results = verifier.verify_end_to_end(sample_data)
    
    # Log results
    logger.info("\n=== Preprocessing Pipeline Verification ===")
    
    if 'pixel_values' in results:
        logger.info("\nPixel Values:")
        for k, v in results['pixel_values'].items():
            logger.info(f"  {k}: {v}")
    
    if 'input_ids' in results:
        logger.info("\nInput IDs:")
        for k, v in results['input_ids'].items():
            logger.info(f"  {k}: {v}")
    
    if 'attention_mask' in results:
        logger.info("\nAttention Mask:")
        for k, v in results['attention_mask'].items():
            logger.info(f"  {k}: {v}")
    
    if 'labels' in results:
        logger.info("\nLabels:")
        for k, v in results['labels'].items():
            logger.info(f"  {k}: {v}")
    
    if results['issues']:
        logger.warning("\nIssues Found:")
        for issue in results['issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("\nNo issues found in preprocessing pipeline.")
    
    return results
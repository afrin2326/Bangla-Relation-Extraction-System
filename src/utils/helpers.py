# ============================================
# src/utils/helpers.py
# Helper Functions
# ============================================

import joblib
import torch
import os
from pathlib import Path

def load_model(model_path, num_classes=7, device='cpu'):
    """
    Load trained model from path
    """
    from src.model.architecture import BanglaREXClassifier
    
    model = BanglaREXClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

def load_encoder(encoder_path):
    """
    Load label encoder from path
    """
    return joblib.load(encoder_path)

def load_tokenizer(tokenizer_path):
    """
    Load tokenizer from path
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path)

def get_project_root():
    """
    Get project root directory
    """
    return Path(__file__).parent.parent.parent

def get_device():
    """
    Get available device (GPU/CPU)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_single(text, model, tokenizer, encoder, device, max_length=128):
    """
    Predict relation for a single text
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    ).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    # Get relation name
    relation = encoder.classes_[prediction]
    
    return {
        'text': text,
        'relation': relation,
        'confidence': confidence,
        'confidence_percentage': confidence * 100,
        'prediction_id': prediction
    }

def predict_batch(texts, model, tokenizer, encoder, device, max_length=128):
    """
    Predict relations for multiple texts
    """
    results = []
    for text in texts:
        result = predict_single(text, model, tokenizer, encoder, device, max_length)
        results.append(result)
    return resultsss
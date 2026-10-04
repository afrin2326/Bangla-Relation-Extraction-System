# ============================================
# src/model/architecture.py
# BanglaBERT Model Architecture for Relation Extraction
# ============================================

import torch
import torch.nn as nn
from transformers import AutoModel

class BanglaREXClassifier(nn.Module):
    """
    BanglaBERT-based classifier for Relation Extraction
    Uses the [CLS] token for classification with dropout regularization
    """
    
    def __init__(self, base_model_name="csebuetnlp/banglabert", num_classes=7, dropout=0.3):
        super().__init__()
        
        # Load pre-trained BanglaBERT
        self.banglabert = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.banglabert.config.hidden_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # Get BanglaBERT outputs
        outputs = self.banglabert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


class BanglaREXClassifierSimplified(nn.Module):
    """
    Simplified version with less overfitting
    """
    
    def __init__(self, base_model_name="csebuetnlp/banglabert", num_classes=7, dropout=0.1):
        super().__init__()
        
        self.banglabert = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.banglabert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.banglabert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
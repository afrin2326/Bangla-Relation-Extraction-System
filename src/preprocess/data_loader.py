# ============================================
# src/preprocess/data_loader.py
# Data Loading and Preprocessing
# ============================================

import pandas as pd
import re
import unicodedata

def fix_bengali_encoding(text):
    """
    Fix Bengali text encoding issues
    """
    if not isinstance(text, str):
        text = str(text)
    try:
        return text.encode('latin1').decode('utf-8')
    except:
        try:
            return text.encode('cp1252').decode('utf-8')
        except:
            return text

def clean_bengali_text(text):
    """
    Clean Bengali text by removing special characters,
    normalizing Unicode, and removing extra spaces
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Fix encoding first
    text = fix_bengali_encoding(text)
    
    # Normalize Unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove special characters except Bengali
    text = re.sub(r'[^\u0980-\u09FF\s\.\,\?\!।॥০-৯]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_formatted_text(text):
    """
    Remove NER/POS tags from formatted text
    """
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r'/[A-Z]+', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def load_dataset(file_path):
    """
    Load dataset from CSV or Excel
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def prepare_text_for_prediction(text):
    """
    Prepare text for prediction
    """
    text = clean_bengali_text(text)
    return text

def load_train_test_data(train_path, test_path):
    """
    Load train and test data
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df
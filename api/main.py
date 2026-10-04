# ============================================
# api/main.py - FastAPI Backend
# ============================================

import sys
import io

# Fix console encoding for Bengali
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import torch
import shutil
import tempfile
import re
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import (
    load_model, load_encoder, load_tokenizer,
    get_device, predict_single, predict_batch
)
from src.preprocess import prepare_text_for_prediction
from src.utils.file_processor import FileProcessor

# ============================================
# Initialize FastAPI App
# ============================================

app = FastAPI(
    title="Bangla Relation Extraction System",
    description="Relation Extraction API for Bengali Text using BanglaBERT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Hardcoded Relation Mapping (CORRECT ORDER)
# ============================================

RELATION_MAPPING = {
    0: 'প্রতিষ্ঠানের অবস্থান',
    1: 'মৃত্যুস্থান', 
    2: 'জন্মস্থান',
    3: 'চলচ্চিত্র পরিচালক',
    4: 'চলচ্চিত্র অভিনেতা',
    5: 'প্রতিষ্ঠাতা',
    6: 'লেখক'
}

HARDCODED_RELATIONS = [
    'প্রতিষ্ঠানের অবস্থান',
    'মৃত্যুস্থান',
    'জন্মস্থান',
    'চলচ্চিত্র পরিচালক',
    'চলচ্চিত্র অভিনেতা',
    'প্রতিষ্ঠাতা',
    'লেখক'
]

def get_relation_name(prediction_id):
    """Get correct relation name from prediction ID"""
    return RELATION_MAPPING.get(prediction_id, f"Unknown ({prediction_id})")

# ============================================
# Load Models
# ============================================

# Get device
device = get_device()
print(f"🔧 Using device: {device}")

# Model paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "banglarex_model" / "model_state.pt"
ENCODER_PATH = BASE_DIR / "encoders" / "relation_encoder.pkl"
TOKENIZER_PATH = BASE_DIR / "models" / "banglarex_model"

# Load model
try:
    model = load_model(str(MODEL_PATH), num_classes=7, device=str(device))
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load tokenizer
try:
    tokenizer = load_tokenizer(str(TOKENIZER_PATH))
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    tokenizer = None

# Load encoder (for reference only)
encoder = None
encoder_classes = HARDCODED_RELATIONS

try:
    encoder = load_encoder(str(ENCODER_PATH))
    print(f"✅ Encoder loaded successfully!")
    try:
        relations = []
        for rel in encoder.classes_:
            try:
                decoded = rel.encode('latin1').decode('utf-8')
                relations.append(decoded)
            except:
                relations.append(rel)
        if relations and len(relations) == 7:
            encoder_classes = relations
            print(f"   Relations: {relations}")
        else:
            print("⚠️ Using hardcoded relations")
            encoder_classes = HARDCODED_RELATIONS
    except Exception as e:
        print(f"⚠️ Could not decode encoder classes: {e}")
        encoder_classes = HARDCODED_RELATIONS
except Exception as e:
    print(f"❌ Error loading encoder: {e}")
    encoder = None
    encoder_classes = HARDCODED_RELATIONS

print(f"✅ Using relations: {encoder_classes}")

# ============================================
# Pydantic Models
# ============================================

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    relation: str
    confidence: float
    confidence_percentage: float

class BatchPredictionRequest(BaseModel):
    texts: List[str]

class BatchPredictionResponse(BaseModel):
    results: List[dict]

# ============================================
# Helper: Extract Entities from Text
# ============================================

def extract_entities_from_text(text: str) -> List[dict]:
    """Extract entity pairs from text using patterns"""
    results = []
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # PATTERN 1: Person Location-এ জন্মগ্রহণ করেন
    pattern1 = r'([^।\n]+?)\s+([\w\s]+[ঃায়ে])\s+জন্মগ্রহণ\s+করেন'
    for match in re.findall(pattern1, text):
        if len(match) >= 2:
            results.append({
                'entity1': match[0].strip(),
                'entity2': match[1].strip(),
                'text': match[0].strip() + ' ' + match[1].strip()
            })
    
    # PATTERN 2: Person 'Movie' ছবিতে অভিনয় করেছেন
    pattern2 = r'([^।\n]+?)\s+[\'"]([^।\n]+?)[\'"]\s+ছবিতে\s+অভিনয়\s+করেছেন'
    for match in re.findall(pattern2, text):
        if len(match) >= 2:
            results.append({
                'entity1': match[0].strip(),
                'entity2': match[1].strip(),
                'text': match[0].strip() + ' ' + match[1].strip()
            })
    
    # PATTERN 3: Person 'Book' লিখেছেন
    pattern3 = r'([^।\n]+?)\s+[\'"]([^।\n]+?)[\'"]\s+লিখেছেন'
    for match in re.findall(pattern3, text):
        if len(match) >= 2:
            results.append({
                'entity1': match[0].strip(),
                'entity2': match[1].strip(),
                'text': match[0].strip() + ' ' + match[1].strip()
            })
    
    # PATTERN 4: Person 'Movie' পরিচালনা করেছেন
    pattern4 = r'([^।\n]+?)\s+[\'"]([^।\n]+?)[\'"]\s+পরিচালনা\s+করেছেন'
    for match in re.findall(pattern4, text):
        if len(match) >= 2:
            results.append({
                'entity1': match[0].strip(),
                'entity2': match[1].strip(),
                'text': match[0].strip() + ' ' + match[1].strip()
            })
    
    # PATTERN 5: Person Company প্রতিষ্ঠা করেছেন
    pattern5 = r'([^।\n]+?)\s+([\w\s]+?)\s+প্রতিষ্ঠা\s+করেছেন'
    for match in re.findall(pattern5, text):
        if len(match) >= 2:
            results.append({
                'entity1': match[0].strip(),
                'entity2': match[1].strip(),
                'text': match[0].strip() + ' ' + match[1].strip()
            })
    
    # PATTERN 6: Person Location-এ জন্ম
    pattern6 = r'([^।\n]+?)\s+([\w\s]+[ঃায়ে])\s+জন্ম'
    for match in re.findall(pattern6, text):
        if len(match) >= 2:
            results.append({
                'entity1': match[0].strip(),
                'entity2': match[1].strip(),
                'text': match[0].strip() + ' ' + match[1].strip()
            })
    
    # Remove duplicates
    unique = []
    seen = set()
    for item in results:
        key = item['entity1'] + '|' + item['entity2']
        if key not in seen:
            seen.add(key)
            unique.append(item)
    
    return unique

# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Health check endpoint"""
    if model is None or tokenizer is None:
        return {
            "status": "Error",
            "message": "Model not loaded properly",
            "model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None,
            "encoder_loaded": encoder is not None
        }
    
    return {
        "status": "Online",
        "message": "Bangla Relation Extraction System is running!",
        "model": "BanglaBERT (csebuetnlp/banglabert)",
        "relations": encoder_classes,
        "device": str(device)
    }

@app.get("/relations")
async def get_relations():
    """Get all available relations"""
    return {
        "relations": encoder_classes,
        "total": len(encoder_classes)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict relation for a single Bengali text"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    text = request.text.strip()
    if not text or len(text) < 2:
        raise HTTPException(status_code=400, detail="Text too short (minimum 2 characters)")
    
    try:
        processed_text = prepare_text_for_prediction(text)
        
        inputs = tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        relation_name = get_relation_name(prediction)
        
        return PredictionResponse(
            text=processed_text,
            relation=relation_name,
            confidence=confidence,
            confidence_percentage=confidence * 100
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_endpoint(request: BatchPredictionRequest):
    """Predict relations for multiple Bengali texts"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        results = []
        for text in request.texts:
            processed_text = prepare_text_for_prediction(text)
            
            inputs = tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(inputs['input_ids'], inputs['attention_mask'])
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            relation_name = get_relation_name(prediction)
            
            results.append({
                'text': processed_text,
                'relation': relation_name,
                'confidence': confidence,
                'confidence_percentage': confidence * 100
            })
        
        return BatchPredictionResponse(results=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# File Upload Endpoint
# ============================================

@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process any file (PDF, TXT, DOCX)
    """
    # Validate file type
    allowed_types = ['.pdf', '.txt', '.docx']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Process file
        result = FileProcessor.process_file(tmp_path)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Print debug info
        print(f"📄 File processed: {file.filename}")
        print(f"   Total sentences: {result['total_sentences']}")
        print(f"   Total entities found: {result['total_entities']}")
        for entity in result.get('entities', []):
            print(f"   - {entity['entity1']} → {entity['entity2']}")
        
        # Extract relations from entities
        predictions = []
        for entity in result.get('entities', []):
            try:
                text = entity['text']
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(inputs['input_ids'], inputs['attention_mask'])
                    probabilities = torch.softmax(outputs, dim=1)
                    prediction = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[0][prediction].item()
                
                relation_name = get_relation_name(prediction)
                
                predictions.append({
                    'entity1': entity['entity1'],
                    'entity2': entity['entity2'],
                    'relation': relation_name,
                    'confidence': f"{confidence * 100:.1f}%"
                })
            except Exception as e:
                print(f"⚠️ Error predicting for {entity}: {e}")
                predictions.append({
                    'entity1': entity['entity1'],
                    'entity2': entity['entity2'],
                    'relation': 'Error',
                    'confidence': '0%'
                })
        
        return {
            'success': True,
            'file_name': file.filename,
            'file_type': result['file_type'],
            'total_sentences': result['total_sentences'],
            'total_entities': result['total_entities'],
            'entities': result['entities'],
            'predictions': predictions
        }
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ============================================
# Direct Text Extraction Endpoint
# ============================================

@app.post("/extract/from-text")
async def extract_from_text(request: dict):
    """
    Direct text extraction (bypass file processing)
    """
    text = request.get('text', '')
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    # Extract entities directly
    entities = extract_entities_from_text(text)
    
    print(f"📝 Text received: {len(text)} characters")
    print(f"🔍 Entities found: {len(entities)}")
    for entity in entities:
        print(f"   - {entity['entity1']} → {entity['entity2']}")
    
    # Predict relations
    predictions = []
    for entity in entities:
        try:
            inputs = tokenizer(
                entity['text'],
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(inputs['input_ids'], inputs['attention_mask'])
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            relation_name = get_relation_name(prediction)
            
            predictions.append({
                'entity1': entity['entity1'],
                'entity2': entity['entity2'],
                'relation': relation_name,
                'confidence': f"{confidence * 100:.1f}%"
            })
        except Exception as e:
            print(f"⚠️ Error: {e}")
            predictions.append({
                'entity1': entity['entity1'],
                'entity2': entity['entity2'],
                'relation': 'Error',
                'confidence': '0%'
            })
    
    return {
        'success': True,
        'total_entities': len(entities),
        'entities': entities,
        'predictions': predictions
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "OK",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "encoder_loaded": encoder is not None,
        "device": str(device),
        "num_classes": len(encoder_classes)
    }

# ============================================
# Run the app
# ============================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
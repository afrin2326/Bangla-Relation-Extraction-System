# Bangla Relation Extraction System 🇧🇩

**Semantic Relation Extraction from Bangla Text using Deep Learning**

Bangla Relation Extraction System is an NLP-based application designed to extract semantic relations between entities in Bengali text using a Transformer-based deep learning model. It provides structured relation predictions with confidence scores through a FastAPI backend, supported by an interactive web interface.

## 🚀 Features

* **Relation Extraction:** Identifies semantic relations between entities in Bengali text.
* **Transformer-Based Model:** Uses a fine-tuned Transformer encoder for contextual understanding.
* **Confidence Scores:** Displays confidence scores alongside predicted relations.
* **REST API:** Exposes prediction functionality through FastAPI.
* **Interactive Web Interface:** Provides a user-friendly interface for testing predictions.
* **Deep Learning:** Uses PyTorch for model implementation and inference.
* **Git LFS Support:** Manages large model files efficiently using Git Large File Storage.

## 🛠️ Technologies Used

| Technology                | Purpose                                   |
| ------------------------- | ----------------------------------------- |
| Python                    | Core programming language                 |
| PyTorch                   | Deep learning and model inference         |
| Hugging Face Transformers | Transformer model ecosystem               |
| FastAPI                   | Backend API                               |
| Uvicorn                   | ASGI server for FastAPI                   |
| Streamlit / Flask         | Interactive web interface                 |
| Pandas                    | Data processing                           |
| Scikit-learn              | Machine learning utilities and evaluation |
| Git LFS                   | Large file storage for model weights      |

## 🏗️ Project Architecture

```text
User
  |
  v
Web Interface
  |
  v
FastAPI Backend
  |
  v
Bangla Text Processing
  |
  v
Transformer-Based Model
  |
  v
Relation Classification Head
  |
  v
Predicted Relations and Confidence Scores

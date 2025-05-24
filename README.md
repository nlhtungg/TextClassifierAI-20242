# Text Classifier AI

A basic web application for text classification using machine learning algorithms with a modern, responsive UI.

## Overview

Text Classifier AI is a Flask-based web service that uses natural language processing and machine learning techniques to classify text into predefined categories. The system employs a Support Vector Machine (SVM) model with Word2Vec embeddings for accurate text categorization.

## Features

- **Real-time classification**: Instantly classify text inputs with response time metrics
- **Modern UI**: Clean, responsive interface with visual feedback and animations
- **Optimized performance**: Pre-loaded models for fast prediction times
- **Multiple algorithms**: Implementation of different classification algorithms (SVM, Naive Bayes, Random Forest, Logistic Regression)
- **Model management**: Automatic model downloading and setup

## Technical Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**: 
  - Word2Vec for text embeddings
  - SVM for classification
  - scikit-learn for ML pipeline
- **Model Storage**: Google Drive integration with gdown for model downloads

## Getting Started

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd TextClassifierAI-20242
```

2. Install the required dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## How It Works

1. The application automatically downloads and sets up the required model files on first run
2. Text input is preprocessed (lowercase conversion, punctuation removal, stopword filtering)
3. The clean text is transformed into vector representations using Word2Vec embeddings
4. The SVM classifier predicts the category of the input text
5. Results are displayed in the UI with the predicted category and processing time

## Project Structure

```
app.py                   # Main Flask application
index.html               # Redirection to templates/index.html
predict_model.py         # Model loading and prediction utilities
predict.py               # Core prediction functionality
requirements.txt         # Python dependencies
setup_models_util.py     # Model download and setup utilities
algorithms/              # Implementation of classification algorithms
  ├── lr.py              # Logistic Regression implementation
  ├── nb.py              # Naive Bayes implementation
  ├── rf.py              # Random Forest implementation
  └── svm.py             # Support Vector Machine implementation
model/                   # Directory for stored model files
  ├── label_encoder.pkl  # Label encoding for categories
  ├── svm_classifier_pipeline.pkl  # Trained SVM model
  └── word2vec_embedding.model     # Word2Vec embeddings
templates/
  └── index.html         # Main UI template
```

## Algorithm Implementations

The project includes custom implementations of several machine learning algorithms:

- **Support Vector Machine**: Linear SVM with gradient descent optimization
- **Naive Bayes**: Gaussian Naive Bayes with probability density function
- **Random Forest**: (Implementation details in rf.py)
- **Logistic Regression**: (Implementation details in lr.py)

## Customization

- You can replace the stopwords list in `predict.py` with your preferred set
- Models can be retrained and updated via the Google Drive folder link in `setup_models_util.py`
- UI customization can be done in `templates/index.html`

## Acknowledgments

- Word2Vec model for text embeddings
- Flask framework for web service
- scikit-learn for machine learning utilities

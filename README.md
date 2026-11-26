
# Real-Time SMS Spam Classification API

## Project Overview

This project deploys a trained Machine Learning model as a REST API
using FastAPI. It predicts whether an SMS message is SPAM or HAM
(NORMAL) and returns prediction probabilities.

## System Architecture

SMS Message (JSON)
       |
       v
   FastAPI Server
       |
       v
   Input Validation
       |
       v
   Text Preprocessing
       |
       v
   TF-IDF Vectorizer
       |
       v
   PyTorch Neural Network
       |
       v
   Sigmoid Probability
       |
       v
   SPAM / HAM Prediction
       |
       v
   JSON Response

## Technology Stack

- Python
- PyTorch
- Scikit-learn
- TF-IDF Vectorizer
- FastAPI
- Pydantic
- Uvicorn
- Pytest
- Docker

## Model Architecture

The trained neural network contains:

- Input layer: 3000 features
- Hidden layer 1: 256 neurons
- Batch Normalization
- ReLU activation
- Dropout
- Hidden layer 2: 128 neurons
- Batch Normalization
- ReLU activation
- Dropout
- Output layer: 1 neuron
- Sigmoid probability output

## Project Structure

sms-spam-api/
|
|-- models/
|   |-- sms_spam_classifier.pth
|   |-- sms_tfidf_vectorizer.pkl
|
|-- tests/
|   |-- test_api.py
|
|-- main.py
|-- requirements.txt
|-- Dockerfile
|-- README.md

## API Endpoints

### 1. Home

GET /

Returns API status information.

### 2. Health Check

GET /health

Returns the health status of the API.

### 3. Prediction

POST /predict

Request:

```json
{
  "message": "Congratulations! You won a free prize!"
}

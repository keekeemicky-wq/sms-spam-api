
import torch
import torch.nn as nn
import joblib
import re
from fastapi import FastAPI
from pydantic import BaseModel, Field

class SpamClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)

app = FastAPI(title="SMS Spam Classifier API")

BASE = "/content/sms-spam-api"

vectorizer = joblib.load(
    BASE + "/models/sms_tfidf_vectorizer.pkl"
)

model = SpamClassifier(3000)

state = torch.load(
    BASE + "/models/sms_spam_classifier.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state)
model.eval()

class MessageInput(BaseModel):
    message: str = Field(min_length=1)

@app.get("/")
def home():
    return {"message": "SMS Spam Classifier API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: MessageInput):
    cleaned = data.message.lower().strip()
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned)

    features = vectorizer.transform([cleaned]).toarray()
    tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)
        probability = torch.sigmoid(output).item()

    label = "SPAM" if probability >= 0.5 else "HAM (NORMAL)"

    return {
        "message": data.message,
        "prediction": label,
        "spam_probability": round(probability, 4),
        "ham_probability": round(1 - probability, 4)
    }

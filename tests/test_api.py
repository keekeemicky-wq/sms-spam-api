
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_spam_prediction():
    response = client.post(
        "/predict",
        json={"message": "You won a free prize! Call now!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["SPAM", "HAM (NORMAL)"]
    assert 0 <= data["spam_probability"] <= 1
    assert 0 <= data["ham_probability"] <= 1
    assert abs(data["spam_probability"] + data["ham_probability"] - 1) < 0.001

def test_normal_prediction():
    response = client.post(
        "/predict",
        json={"message": "Hello, how are you?"}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_empty_message():
    response = client.post(
        "/predict",
        json={"message": ""}
    )
    assert response.status_code == 422

def test_missing_message():
    response = client.post("/predict", json={})
    assert response.status_code == 422

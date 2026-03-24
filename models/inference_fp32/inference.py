"""
SageMaker inference script for FP32 PyTorch models (Variant A: BERT, Variant C: DistilBERT).
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def model_fn(model_dir):
    """Load tokenizer and model from model_dir."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}


def predict_fn(data, model_dict):
    """Run inference on a single input."""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    text = data["inputs"]

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)
    predicted_id = int(torch.argmax(probs, dim=-1).item())
    confidence = round(float(probs[0, predicted_id].item()), 4)

    predicted_label = model.config.id2label.get(predicted_id, str(predicted_id))

    return {
        "predicted_label": predicted_label,
        "predicted_id": predicted_id,
        "confidence": confidence,
    }

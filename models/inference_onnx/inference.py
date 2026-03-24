"""
SageMaker inference script for ONNX INT8 model (Variant B: BERT-INT8).
"""
import json
import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def model_fn(model_dir):
    """Load tokenizer, ONNX session, and id2label mapping from model_dir."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    onnx_path = os.path.join(model_dir, "model.onnx")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Load id2label from config.json
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    id2label = {int(k): v for k, v in config.get("id2label", {}).items()}

    return {
        "session": session,
        "tokenizer": tokenizer,
        "id2label": id2label,
    }


def predict_fn(data, model_dict):
    """Run inference on a single input using the ONNX session."""
    session = model_dict["session"]
    tokenizer = model_dict["tokenizer"]
    id2label = model_dict["id2label"]

    text = data["inputs"]

    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    input_feed = {k: v for k, v in inputs.items()}
    outputs = session.run(["logits"], input_feed)
    logits = outputs[0]

    # Manual softmax with numpy
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    predicted_id = int(np.argmax(probs, axis=-1)[0])
    confidence = round(float(probs[0, predicted_id]), 4)

    predicted_label = id2label.get(predicted_id, str(predicted_id))

    return {
        "predicted_label": predicted_label,
        "predicted_id": predicted_id,
        "confidence": confidence,
    }

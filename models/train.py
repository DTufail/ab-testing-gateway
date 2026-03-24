"""
Training script for BANKING77 intent classification.
Supports both bert-base-uncased (Variant A) and distilbert-base-uncased (Variant C).
Parametrised entirely by environment variables.
"""
import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score

# --- Hyperparameters from environment ---
MODEL_NAME   = os.environ.get("MODEL_NAME", "bert-base-uncased")
NUM_EPOCHS   = int(os.environ.get("NUM_EPOCHS", "3"))
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "./output")
SMOKE_TEST   = os.environ.get("SMOKE_TEST", "0") == "1"

if SMOKE_TEST:
    print("*** SMOKE TEST MODE — using 50 train / 20 eval examples, 1 epoch ***")
    NUM_EPOCHS = 1

os.makedirs(SM_MODEL_DIR, exist_ok=True)

# --- Load dataset ---
print(f"Loading PolyAI/banking77 dataset...")
dataset = load_dataset("PolyAI/banking77", revision="refs/convert/parquet")

if SMOKE_TEST:
    dataset["train"] = dataset["train"].select(range(50))
    dataset["test"]  = dataset["test"].select(range(20))

num_labels = dataset["train"].features["label"].num_classes
label_names = dataset["train"].features["label"].names

# --- Tokenizer ---
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    # Do NOT pad to max_length here — DataCollatorWithPadding handles dynamic padding per batch
    return tokenizer(batch["text"], truncation=True, max_length=128)


print("Tokenizing dataset...")
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
# Do NOT call set_format("torch") — incompatible with DataCollatorWithPadding; Trainer handles it

# --- Model ---
print(f"Loading model {MODEL_NAME} with {num_labels} labels...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label={i: name for i, name in enumerate(label_names)},
    label2id={name: i for i, name in enumerate(label_names)},
)

# --- Data collator ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# --- Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


# --- Training arguments ---
training_args = TrainingArguments(
    output_dir=SM_MODEL_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=not SMOKE_TEST,  # CPU (smoke test) doesn't support fp16
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# --- Save model and tokenizer ---
print(f"Saving model and tokenizer to {SM_MODEL_DIR}...")
trainer.save_model(SM_MODEL_DIR)
tokenizer.save_pretrained(SM_MODEL_DIR)

print("Training complete.")

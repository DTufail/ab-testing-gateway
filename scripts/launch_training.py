"""
Launch SageMaker training jobs for BANKING77 intent classification.
Supports launching BERT (Variant A), DistilBERT (Variant C), or both.
"""
import argparse
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from sagemaker.huggingface import HuggingFace


def parse_args():
    parser = argparse.ArgumentParser(description="Launch SageMaker training jobs")
    parser.add_argument(
        "--variant",
        choices=["bert", "distilbert", "all"],
        default="all",
        help="Which variant(s) to train (default: all)",
    )
    return parser.parse_args()


def build_bert_estimator():
    return HuggingFace(
        entry_point="train.py",
        source_dir="models/",
        role=config.SAGEMAKER_ROLE,
        instance_type=config.INSTANCE_TYPE_TRAIN,
        instance_count=1,
        transformers_version=config.TRANSFORMERS_VERSION,
        pytorch_version=config.PYTORCH_VERSION,
        py_version=config.PY_VERSION,
        hyperparameters={},
        environment={
            "MODEL_NAME": "bert-base-uncased",
            "NUM_EPOCHS": "3",
        },
        output_path=f"s3://{config.S3_BUCKET}/models/bert-fp32/",
    )


def build_distilbert_estimator():
    return HuggingFace(
        entry_point="train.py",
        source_dir="models/",
        role=config.SAGEMAKER_ROLE,
        instance_type=config.INSTANCE_TYPE_TRAIN,
        instance_count=1,
        transformers_version=config.TRANSFORMERS_VERSION,
        pytorch_version=config.PYTORCH_VERSION,
        py_version=config.PY_VERSION,
        hyperparameters={},
        environment={
            "MODEL_NAME": "distilbert-base-uncased",
            "NUM_EPOCHS": "3",
        },
        output_path=f"s3://{config.S3_BUCKET}/models/distilbert-fp32/",
    )


def main():
    args = parse_args()

    print(f"Launching training jobs for variant: {args.variant}")
    print(f"S3 bucket: {config.S3_BUCKET}")
    print(f"SageMaker role: {config.SAGEMAKER_ROLE}")
    print(f"Instance type: {config.INSTANCE_TYPE_TRAIN}\n")

    if args.variant in ("bert", "all"):
        print("Starting BERT (Variant A) training job...")
        bert_estimator = build_bert_estimator()
        bert_estimator.fit(wait=False)
        print(f"BERT job name: {bert_estimator.latest_training_job.name}")

    if args.variant in ("distilbert", "all"):
        print("Starting DistilBERT (Variant C) training job...")
        distilbert_estimator = build_distilbert_estimator()
        distilbert_estimator.fit(wait=False)
        print(f"DistilBERT job name: {distilbert_estimator.latest_training_job.name}")

    print("\nMonitor jobs at: https://console.aws.amazon.com/sagemaker/home#/jobs")


if __name__ == "__main__":
    main()

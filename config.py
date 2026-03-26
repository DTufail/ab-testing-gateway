# config.py
import os

AWS_REGION        = "us-east-1"
AWS_ACCOUNT_ID    = os.environ.get("AWS_ACCOUNT_ID", "011190986627")
S3_BUCKET         = "ab-gateway-artifacts"
SAGEMAKER_ROLE    = os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::011190986627:role/service-role/AmazonSageMaker-ExecutionRole-20260324T125083")

# S3 paths — original model weights (raw training output, weights only)
S3_BERT_FP32  = f"s3://{S3_BUCKET}/models/bert-fp32/huggingface-pytorch-training-2026-03-24-14-48-08-677/output/model.tar.gz"
S3_BERT_INT8  = f"s3://{S3_BUCKET}/models/bert-int8/model.tar.gz"
S3_DISTILBERT = f"s3://{S3_BUCKET}/models/distilbert-fp32/huggingface-pytorch-training-2026-03-24-15-02-31-971/output/model.tar.gz"

# S3 paths — bundled archives (model weights + code/inference.py in one tar.gz)
# The HF DLC finds inference.py at /opt/ml/model/code/inference.py automatically.
# No SAGEMAKER_SUBMIT_DIRECTORY or SAGEMAKER_PROGRAM env vars needed.
S3_BERT_FP32_BUNDLED  = f"s3://{S3_BUCKET}/models/bert-fp32-bundled/model.tar.gz"
S3_DISTILBERT_BUNDLED = f"s3://{S3_BUCKET}/models/distilbert-bundled/model.tar.gz"

# SageMaker training
INSTANCE_TYPE_TRAIN   = "ml.g4dn.xlarge"
TRANSFORMERS_VERSION  = "4.26"
PYTORCH_VERSION       = "1.13"
PY_VERSION            = "py39"

# HuggingFace DLC image for inference (CPU, us-east-1)
# Update URI if deploying in a different region:
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
HF_DLC_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04"
)

# SageMaker endpoint
INSTANCE_TYPE_INFER = "ml.m5.large"
ENDPOINT_NAME_PREFIX = "ab-gateway-endpoint"

# Variant names — used in both deploy and validate scripts
VARIANT_A = "VariantA-BERT-FP32"
VARIANT_B = "VariantB-BERT-INT8"
VARIANT_C = "VariantC-DistilBERT"
ALL_VARIANTS = [VARIANT_A, VARIANT_B, VARIANT_C]

# Model Registry group names
MODEL_GROUPS = {
    VARIANT_A: "BERT-FP32-Banking77",
    VARIANT_B: "BERT-INT8-Banking77",
    VARIANT_C: "DistilBERT-FP32-Banking77",
}

# Traffic weights for initial deployment (must sum to same total — SageMaker normalises)
INITIAL_WEIGHTS = {
    VARIANT_A: 6,   # 60%
    VARIANT_B: 2,   # 20%
    VARIANT_C: 2,   # 20%
}

# Expected accuracy floor for golden set validation in CI/CD (Phase 4)
MIN_ACCURACY_BERT       = 0.90
MIN_ACCURACY_DISTILBERT = 0.88

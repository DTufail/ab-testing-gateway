# Multi-Model A/B Testing Gateway — Phase 1

A portfolio project demonstrating production-grade MLOps on AWS: three BERT-family intent classifiers deployed behind a single SageMaker endpoint with configurable traffic splitting. The system targets the [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) dataset (77-class customer support intent classification) and is designed to be extended with automated traffic shifting, drift detection, and CI/CD gates in later phases.

## Architecture

| Variant | Model | Format | Role | Initial Traffic |
|---|---|---|---|---|
| `VariantA-BERT-FP32` | `bert-base-uncased` fine-tuned | PyTorch FP32 | Accuracy baseline | 60% |
| `VariantB-BERT-INT8` | Same BERT, ONNX INT8 (Optimum) | ONNX INT8 | Speed/cost tradeoff | 20% |
| `VariantC-DistilBERT` | `distilbert-base-uncased` fine-tuned | PyTorch FP32 | Budget option | 20% |

All three variants live on a single SageMaker endpoint. Downstream phases adjust weights based on live latency and accuracy metrics.

---

## Prerequisites

- **AWS CLI** configured with credentials (`aws configure`)
- **IAM role** with SageMaker full access, S3 read/write, and ECR pull permissions
- **S3 bucket** created: `ab-gateway-artifacts-<your-account-id>`
  ```bash
  aws s3 mb s3://ab-gateway-artifacts-<your-account-id> --region us-east-1
  ```
- Python 3.9+ on your local machine (no GPU required — training runs on SageMaker)

---

## Quickstart — Phase 1

```bash
pip install -r requirements-local.txt

# Set your AWS config
export AWS_ACCOUNT_ID=your-account-id
export SAGEMAKER_ROLE_ARN=arn:aws:iam::your-account-id:role/your-role

# Prepare golden test set (used in Phase 4 CI/CD)
python models/prepare_golden_set.py

# Launch training (runs on SageMaker ml.g4dn.xlarge, ~35 min)
python scripts/launch_training.py --variant all

# After training completes, download BERT checkpoint from S3 locally
aws s3 cp s3://$S3_BUCKET/models/bert-fp32/output/<job-name>/output/model.tar.gz ./bert-fp32-raw.tar.gz
mkdir bert-fp32-finetuned && tar -xzf bert-fp32-raw.tar.gz -C bert-fp32-finetuned/

# Quantize BERT → ONNX INT8 and package for SageMaker
./scripts/quantize_and_package.sh ./bert-fp32-finetuned $S3_BUCKET

# Package FP32 models with inference scripts and upload
./scripts/package_fp32.sh bert $S3_BUCKET
./scripts/package_fp32.sh distilbert $S3_BUCKET

# Register all three variants in SageMaker Model Registry
python scripts/register_models.py

# Deploy single multi-variant endpoint (takes ~10 min)
python scripts/deploy_endpoint.py

# Validate all three variants return correct predictions
python scripts/validate_endpoint.py

# Capture baseline latency benchmark
python scripts/measure_latency.py

# Tear down endpoint when done to avoid charges
aws sagemaker delete-endpoint --endpoint-name $(cat .endpoint_name)
```

> **Tip:** Training job names are printed by `launch_training.py`. Monitor them at
> https://console.aws.amazon.com/sagemaker/home#/jobs

---

## Cost Estimate

| Resource | Duration | Cost |
|---|---|---|
| 2× `ml.g4dn.xlarge` training jobs | ~35 min each | ~$0.95 |
| 1× `ml.m5.large` endpoint | 30 min (validate + benchmark) | ~$0.05 |
| S3 storage | ~500 MB | ~$0.01 |
| **Total** | | **~$1.60** |

Costs assume `us-east-1` on-demand pricing. Tear down the endpoint immediately after benchmarking to avoid ongoing charges.

---

## What Phase 2 needs

- Endpoint name (stored in `.endpoint_name` after deployment)
- Variant names: `VariantA-BERT-FP32`, `VariantB-BERT-INT8`, `VariantC-DistilBERT`
- `benchmarks/phase1_baseline.json` — latency baselines for comparison
- `models/golden_test_set.jsonl` — held-out examples for accuracy gating

---

## Repo structure

```
ab-testing-gateway/
├── models/
│   ├── train.py                    # Shared training script (BERT + DistilBERT)
│   ├── quantize.py                 # BERT → ONNX INT8 via Optimum
│   ├── prepare_golden_set.py       # Build Phase 4 golden test set
│   ├── inference_fp32/
│   │   ├── inference.py            # SageMaker serve script (PyTorch FP32)
│   │   └── requirements.txt
│   └── inference_onnx/
│       ├── inference.py            # SageMaker serve script (ONNX runtime)
│       └── requirements.txt
├── scripts/
│   ├── launch_training.py          # Submit SageMaker training jobs
│   ├── quantize_and_package.sh     # Quantize + package INT8 artifact
│   ├── package_fp32.sh             # Inject inference code + repackage FP32
│   ├── register_models.py          # Register in SageMaker Model Registry
│   ├── deploy_endpoint.py          # Create multi-variant endpoint
│   ├── validate_endpoint.py        # Smoke-test all three variants
│   └── measure_latency.py          # p50/p95/p99 latency benchmark
├── benchmarks/
│   └── phase1_baseline.json        # Written by measure_latency.py
├── config.py                       # Single source of truth for all config
└── requirements-local.txt
```

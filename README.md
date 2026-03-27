![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![AWS CDK](https://img.shields.io/badge/AWS-CDK-orange) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

# Multi-Model A/B Testing Gateway

A production-grade MLOps system that routes banking intent classification requests across three BERT-family variants on a single SageMaker endpoint, with a Lambda-based traffic controller that rebalances per-variant weights every 15 minutes on live performance data and a CDK-defined canary pipeline that validates and promotes new model versions automatically.

The core challenge isn't deploying multiple variants to one endpoint — SageMaker handles that natively. The hard parts are deciding how much traffic to send to each variant based on live signal, detecting when a variant is underperforming, rebalancing without human intervention, and safely promoting a new model version without exposing all traffic to an unvalidated change.

This system solves all four. The traffic controller reads confidence scores, latency, and error rates from two CloudWatch namespaces every 15 minutes and applies a dampened scoring formula (`confidence × (1/normalised_latency) × (1 − error_rate)`) to rebalance weights, with a 5% floor per variant and a 10% error-rate kill-switch. The canary pipeline gates every new model version through a CodeBuild accuracy check against a 200-example golden set, shifts 10% of traffic to the candidate, waits 15 minutes, then reads CloudWatch metrics to decide whether to promote or roll back.

The classification task is BANKING77 — 77-class banking intent detection, 13,083 training examples. It is a clean supervised benchmark with enough classes to stress-test routing and accuracy gating across meaningfully different model architectures.

---

## Architecture

**Request flow**
```
Client
  └─► API Gateway
        └─► Lambda (ab-gateway-router)
              ├─► DynamoDB (ab-gateway-routing-config)  [reads weights + strategy]
              └─► SageMaker Endpoint
                    ├─► VariantA-BERT-FP32   (~42% weight)
                    ├─► VariantB-BERT-INT8   (~44% weight)
                    └─► VariantC-DistilBERT  (~14% weight)
```

**Canary deployment flow**
```
Model Registry approval
  └─► EventBridge (ab-gateway-model-approved)
        └─► Lambda (ab-gateway-canary-deploy)
              └─► Step Functions (ab-gateway-canary-workflow)
                    ├─► CodeBuild: validate accuracy on 200-example golden set (gate: ≥80%)
                    ├─► Shift 10% traffic to new variant
                    ├─► Wait 15 minutes (native SFN Wait state)
                    ├─► Read CloudWatch metrics (error rate + p95 latency)
                    └─► Promote to 100%  or  rollback
```

**Traffic controller (separate cycle)**
```
EventBridge (15-min schedule)
  └─► Lambda (ab-gateway-traffic-controller)
        ├─► Read ABGateway + AWS/SageMaker CloudWatch namespaces
        ├─► Score: confidence × (1/norm_latency) × (1 − error_rate)
        ├─► Apply 0.30 dampening, enforce 5% floor per variant
        └─► UpdateEndpointWeightsAndCapacities
```

Step Functions Standard Workflow was chosen over Express because Lambda's maximum execution time is 15 minutes, which leaves zero margin for a 15-minute baking wait plus validation time. Express Workflows cap at 5 minutes — also insufficient. `UpdateEndpointWeightsAndCapacities` was chosen over `UpdateEndpoint` because `UpdateEndpoint` provisions a second instance fleet via blue/green swap, takes roughly 10 minutes, and doubles instance cost during the transition. `UpdateEndpointWeightsAndCapacities` shifts weights on the existing fleet in seconds at no additional cost and preserves the traffic controller's accumulated rebalancing history.

---

## Models

| Variant | Base model | Format | Accuracy | Native p50 |
|---|---|---|---|---|
| VariantA-BERT-FP32 | bert-base-uncased | PyTorch FP32 | 91.27% | 278ms |
| VariantB-BERT-INT8 | bert-base-uncased | ONNX INT8 | ~91% | 180ms |
| VariantC-DistilBERT | distilbert-base-uncased | PyTorch FP32 | 89.45% | 134ms |

All three fine-tuned on BANKING77 for 6 epochs with AdamW and a 10% warmup ratio. INT8 is a post-training dynamic quantization of the FP32 BERT checkpoint via `ORTQuantizer`, producing a 3.97x compression ratio (418MB → 105MB) with negligible accuracy loss.

---

## Infrastructure

- **Routing** — four strategies: weighted random, header-pinned (`X-Variant`), least-latency, shadow mode; all responses include the active variant name and strategy; metrics emitted via EMF to the `ABGateway` CloudWatch namespace with no direct `put_metric_data` calls in the hot path.
- **Traffic controller** — 15-minute EventBridge schedule, dampened scoring formula with 0.30 factor, 5% per-variant floor, 10% error-rate kill-switch, writes updated weights to DynamoDB and calls `UpdateEndpointWeightsAndCapacities`.
- **CI/CD** — self-mutating CDK Pipeline (`ab-gateway-infra-pipeline`) for infrastructure changes; model promotion triggered separately by Model Registry approval events flowing through EventBridge to a Step Functions Standard Workflow canary with CodeBuild validation gate and 15-minute baking period; 8/8 integration tests passing.
- **Observability** — CloudWatch dashboard (`ABGateway-Dashboard`), three alarms (`HighErrorRate`, `HighP95Latency`, `ControllerHeartbeatMissed`), SNS notifications for canary outcomes.

---

## Repository structure

```
ab-testing-gateway/
├── models/
│   ├── train.py                  # Fine-tuning script (BANKING77, bert/distilbert)
│   ├── quantize.py               # ONNX export + INT8 quantization via ORTQuantizer
│   ├── inference_fp32/           # PyTorch inference handler for SageMaker
│   └── inference_onnx/           # ONNX Runtime inference handler for SageMaker
├── lambda/
│   ├── router/                   # Routing Lambda: strategy dispatch + EMF metrics
│   └── traffic_controller/       # Weight rebalancing Lambda
├── cdk/
│   ├── app.py                    # CDK entry point
│   ├── cdk.json                  # All config: account, region, thresholds, model groups
│   ├── stacks/                   # Pipeline, ValidationStack, CanaryStack
│   └── lambda/                   # Six Step Functions worker Lambdas
├── scripts/
│   ├── approve_model_version.py  # Trigger canary by approving a Model Registry version
│   └── test_pipeline.py          # 8-test integration suite
├── benchmarks/
│   ├── golden_test_set.jsonl     # 200-example BANKING77 validation set
│   └── phase1_sagemaker_native_latency.json
├── FINDINGS.md                   # Full technical narrative: failures, fixes, benchmark analysis
└── README.md
```

---

## Setup

Deploying this system requires an AWS account, CDK bootstrapping, and a GitHub connection wired through AWS CodeConnections. The endpoint runs on `ml.m5.large` (CPU); training runs on `ml.g4dn.xlarge`. At full deployment the endpoint costs approximately $83/month — tear it down when not in use.

**Prerequisites**
- AWS account with IAM permissions for SageMaker, Lambda, Step Functions, CodeBuild, EventBridge, DynamoDB, CDK
- AWS CDK 2.170.0: `npm install -g aws-cdk@2.170.0`
- Python 3.12, Docker (for SageMaker packaging)
- Configured AWS credentials: `aws configure`

**One-time setup**
```bash
# 1. Bootstrap CDK
cdk bootstrap aws://ACCOUNT_ID/us-east-1

# 2. Create a GitHub CodeConnections connection in the AWS console:
#    Developer Tools → Connections → Create connection → GitHub
#    Paste the resulting ARN into cdk/cdk.json as "connection_arn"

# 3. Install CDK dependencies
cd cdk && pip install -r requirements.txt

# 4. Verify synthesis
cdk synth
```

**Deploy**
```bash
# Push to GitHub first — the pipeline reads directly from the repo
git push origin main

# Deploy the pipeline stack (self-mutating after this; run once only)
cdk deploy ABGatewayPipelineStack
```

**Train and register models**
```bash
# Train on SageMaker (~45 min per model on ml.g4dn.xlarge)
python scripts/launch_training.py --variant bert
python scripts/launch_training.py --variant distilbert

# Quantize BERT FP32 → ONNX INT8
python models/quantize.py

# Package models with inference scripts bundled inside the archive
bash scripts/package_fp32.sh bert ab-gateway-artifacts <s3-raw-bert-uri>
bash scripts/package_fp32.sh distilbert ab-gateway-artifacts <s3-raw-distilbert-uri>

# Register all three variants in SageMaker Model Registry
python scripts/register_models.py
```

**Trigger a canary deployment**
```bash
# Approve a model version — fires EventBridge → canary pipeline
python scripts/approve_model_version.py --group VariantB-BERT-INT8-ModelGroup
```

**Teardown**
```bash
# Delete all stacks and stop endpoint billing
cdk destroy --all
```

---

## Key findings

The full technical narrative — including the smoke test incident, three successive dependency failures during retraining, Phase 4 operational failures, and the three-layer latency breakdown — is in [FINDINGS.md](./FINDINGS.md).

The headline finding: INT8 quantization delivers a genuine 1.55x speedup at the container inference level (278ms → 180ms native p50). At the endpoint level, ~1,200ms of fixed infrastructure overhead from API Gateway, Lambda routing, and SageMaker request serialization dwarfs the compute savings — the speedup is real but invisible to the client without architectural changes to reduce the infrastructure layers.

---

## License

MIT

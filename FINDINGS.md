# FINDINGS.md

## 1. Overview

This project is a multi-model A/B testing gateway deployed on AWS SageMaker that serves three BERT-based intent classification variants behind a single endpoint, with a traffic controller that continuously rebalances per-variant traffic weights based on live latency and accuracy signals. The classification task is BANKING77 — 77-class banking intent detection across 3,080 evaluation examples. The gateway is backed by a CDK-defined CI/CD canary pipeline that promotes new model versions through a validation accuracy gate (≥80% on a 200-example golden test set) and a 15-minute baking period before full promotion. The project covers end-to-end MLOps engineering: fine-tuning, post-training quantization, SageMaker multi-variant deployment, Lambda-based routing with four traffic strategies, CloudWatch metric-driven rebalancing, and automated model promotion via EventBridge and Step Functions.

---

## 2. The Problem With the First Training Run

The training script includes a `SMOKE_TEST=1` flag designed for pipeline validation during development. When set, it caps the training dataset at 50 examples and runs only 2 gradient steps — enough to confirm the training loop executes without error, not enough to produce a useful model. The flag was left enabled when the production training job was submitted to SageMaker. The job completed in a fraction of the expected time, reported no errors, and uploaded a model artifact that looked externally identical to one produced by a full training run.

What was actually deployed was a BERT model that had seen 50 of the 10,003 available training examples. Its 73% accuracy on BANKING77 came almost entirely from BERT's pre-training on BookCorpus and Wikipedia, not from task-specific fine-tuning. For a 77-class problem, 73% is close to what zero-shot BERT inference produces without any fine-tuning at all. DistilBERT had a separate but related problem: it was trained on the full dataset but for only 3 epochs. The training curves showed it was still converging at epoch 3, and it reached 75.6% — below both the 80% canary gate and the 88% DistilBERT accuracy floor the project had set for itself. Every Phase 1 benchmark was measured against these undertrained models. The latency numbers were real measurements of real SageMaker endpoints, but of models that had not been properly trained. This kind of error — a development flag left active in a production submission — is common. The canary gate caught it only indirectly; the 80% threshold existed for the CI/CD pipeline, not as a runtime check on the initial deployment. Both models were retrained from scratch with `SMOKE_TEST` unset and epochs set to 6.

---

## 3. The Dependency Failures During Retraining

Fixing the training environment required resolving three successive failures, each revealed only after the previous one was addressed.

The first failure: the `requirements.txt` pinned `transformers==4.41.0` for the SageMaker HuggingFace DLC container, which runs PyTorch 1.13. `transformers 4.39.0` introduced a reference to `torch.optim.LRScheduler`, a class that was only publicly exposed in PyTorch 2.0. On PyTorch 1.13 the import crashes immediately with an `AttributeError`. The fix was to downgrade to `transformers==4.38.1`, the last release before the breaking import was introduced.

After that downgrade, a second failure appeared. The HuggingFace Trainer uses `accelerate` internally for mixed precision and DataLoader management. `accelerate` was passing `prefetch_factor` to PyTorch's DataLoader. In PyTorch 1.13, `prefetch_factor` is only a valid argument when `num_workers > 0`. With the default `num_workers=0`, PyTorch raises a `ValueError`. Adding `dataloader_num_workers=1` to the `TrainingArguments` appeared to resolve it.

With `num_workers=1` set, a third failure surfaced: `accelerate` passed `prefetch_factor=None` to the DataLoader. PyTorch 1.13's internal assertion `prefetch_factor > 0` cannot compare `None` with an integer and raises a `TypeError`. The clean fix was to remove all version pins from `requirements.txt` entirely and let the SageMaker container use its native, tested dependency stack — PyTorch 1.13, `transformers 4.26`, and the bundled `accelerate` version. AWS built and validated these three together. With no `requirements.txt` overrides, the training job completed successfully on the next run. The lesson is that when deploying to a managed container, overriding its native stack with `requirements.txt` pins is fragile. The container's tested stack exists for a reason, and fighting it is not worth the incremental version difference.

---

## 4. The Quantization Pipeline

BERT FP32 was exported to ONNX format and then quantized to INT8 using ONNX Runtime's `ORTQuantizer` with `QuantizationConfig` dynamic quantization. The input was a 418MB FP32 ONNX model. The output was a 105MB INT8 ONNX model, a 3.97x compression ratio. Inspection of the quantized graph confirmed 98 `MatMulInteger` ops replacing the standard `MatMul` ops, and 154 quantized weight tensors.

Two issues appeared during this process. The first: `ORTModelForSequenceClassification.from_pretrained` was called with a `./local_path` relative path using the `./` prefix. Newer versions of `optimum` attempt to validate the path string as a HuggingFace Hub repository ID before checking the filesystem. The `./` prefix fails Hub ID validation and raises an error before the filesystem is ever consulted. Resolving the path to an absolute string before passing it to `from_pretrained` fixed the issue.

The second: `transformers` blocks `torch.load` on `.bin` weight files for PyTorch versions below 2.6, citing CVE-2025-32434, a pickle-based deserialization vulnerability. PyTorch 2.6 was not available in the local environment. The fix was to load the weights directly with `torch.load(..., weights_only=False)` — bypassing the transformers-level check, which applies only when loading through the transformers API — and then save the weights with `safetensors`. When weights are in `safetensors` format, transformers loads them via a separate, safe code path and the CVE check does not apply.

---

## 5. Phase 4 Operational Failures

Before the final integration session, three separate failures in the CI/CD pipeline needed to be resolved.

The EventBridge rule `ab-gateway-model-approved` existed in AWS, was in ENABLED state, and its event pattern correctly matched `SageMaker Model Package State Change` events with `ModelApprovalStatus=Approved` across the three model package groups. Its Targets list was null. The CDK had defined the rule but had not persisted the Lambda target association correctly. Every model approval event fired, matched the pattern, and was silently dropped. Nothing downstream ever received it. This went undetected because EventBridge does not surface delivery failures on the rule itself — they appear only in the target's dead-letter queue or CloudWatch metrics, neither of which was being monitored.

Once the target was manually wired, three model approvals were submitted simultaneously for all three variants. Three Step Functions executions launched in parallel, and all three attempted to call `UpdateEndpointWeightsAndCapacities` on the same endpoint at the same time. SageMaker rejected all three with `ValidationException: Cannot update in-progress endpoint`. All three executions failed. The correct operational procedure is to approve one model version at a time and wait for the canary execution to complete before approving the next.

After resolving the race condition, the router Lambda still could not invoke the new endpoint. Six Lambda functions had their `ENDPOINT_NAME` environment variable set to the endpoint name from a previous deployment. More critically, the IAM inline policy on the router Lambda's execution role had the old endpoint ARN hardcoded as a resource literal, which meant the Lambda was prohibited by IAM from calling `invoke_endpoint` on the new endpoint regardless of what the environment variable said. Both were corrected: environment variables updated across all six functions, and the IAM policy resource updated to a wildcard pattern `ab-gateway-endpoint-*` that survives future redeployments without requiring a manual policy update.

A fourth issue appeared during the first smoke test after all three fixes: the CodeBuild project's `ENDPOINT_NAME` environment variable was also frozen at the old endpoint name. This was a separate store from the Lambda env vars — CodeBuild project configuration is set at CDK synth time from `benchmarks/endpoint_name.txt`, and the project had not been redeployed since the endpoint was replaced. The validation script ran 200 inferences against a deleted endpoint, received 200 `ValidationError: Endpoint not found` responses, scored 0.0% accuracy, and aborted. The fix was an `aws codebuild update-project` call to set the correct endpoint name, followed by a code push so the CDK becomes the authoritative source on the next pipeline run. With all four stale references resolved, execution `canary-VariantB-BERT-INT8-1774605948` ran the full state machine successfully: CodeBuild validation passed, weight shifted to 10% canary, 15-minute baking period completed, health check passed, and VariantB-BERT-INT8 was promoted.

---

## 6. Benchmark Results and What They Mean

The latency data separates into three layers, and the relationship between them is the most important finding in this project.

At the container inference level — SageMaker's native `ModelLatency` metric from CloudWatch, which measures only the time inside the serving container — the differences between variants are meaningful:

| Variant | p50 | p95 | Speedup vs FP32 |
|---|---|---|---|
| BERT FP32 | 278ms | 291ms | baseline |
| BERT INT8 | 180ms | 197ms | 1.55x |
| DistilBERT | 134ms | 149ms | 2.08x |

At the end-to-end level — measured from a boto3 `invoke_endpoint` call to response, across 100 requests — the picture changes:

| Variant | p50 | p95 | p99 | mean |
|---|---|---|---|---|
| BERT FP32 | 1,664ms | 2,249ms | 2,614ms | 1,732ms |
| BERT INT8 | 1,481ms | 1,745ms | 1,892ms | 1,491ms |
| DistilBERT | 1,432ms | 1,717ms | 1,802ms | 1,453ms |

The difference between the two layers is fixed infrastructure overhead: HTTPS round trip, API Gateway, Lambda routing, and SageMaker request serialization/deserialization. Subtracting native p50 from end-to-end p50 gives ~1,386ms for FP32, ~1,301ms for INT8, and ~1,298ms for DistilBERT — consistently ~1,100–1,200ms regardless of which model variant handles the request.

INT8 quantization delivers a genuine 1.55x speedup at the inference container level (278ms → 180ms). DistilBERT delivers 2.08x (278ms → 134ms). But both speedups are largely invisible at the endpoint level because ~1,200ms of fixed overhead dwarfs 100–150ms of variable compute savings. This is not a model failure. It is an accurate description of where cost lives in a SageMaker real-time inference deployment. Exposing the compute speedup at the client level requires reducing infrastructure overhead — a VPC endpoint for SageMaker removes the internet round trip, or moving to SageMaker serverless inference eliminates the API Gateway layer entirely.

On accuracy, all three variants pass the 80% canary gate comfortably:

| Variant | Accuracy | Notes |
|---|---|---|
| BERT FP32 | 91.27% | 6-epoch fine-tune, full 10,003-example training set |
| BERT INT8 | ~91% | <1% degradation from `ORTQuantizer` dynamic quantization |
| DistilBERT | 89.45% | 6-epoch fine-tune, 66M vs 110M parameters |

The 1.82 percentage point gap between DistilBERT (89.45%) and BERT FP32 (91.27%) is the capacity cost of a 40% smaller model. Both are well above the 80% gate and the 88% DistilBERT floor. The traffic controller exists in part to detect if that gap widens on live traffic — live weights at project completion were INT8=44%, FP32=42%, DistilBERT=14%, reflecting INT8's combination of near-FP32 accuracy and 1.55x lower inference latency.

---

## 7. System Architecture Summary

The endpoint is a single SageMaker multi-variant endpoint on an `ml.m5.large` CPU instance. Three variants are registered: `VariantA-BERT-FP32` at 60% initial traffic weight, `VariantB-BERT-INT8` at 20%, and `VariantC-DistilBERT` at 20%. All three are registered in SageMaker Model Registry with dedicated Model Package Groups.

The routing layer (Phase 2) places API Gateway in front of the Lambda router `ab-gateway-router`, which reads per-variant configuration from DynamoDB (`ab-gateway-routing-config`) and forwards to SageMaker. Four routing strategies are implemented: weighted random, header-pinned (client sends `X-Variant` header), least-latency, and shadow mode. Every response includes the variant name and active strategy. Metrics are emitted via EMF (Embedded Metrics Format) to the `ABGateway` CloudWatch namespace — no direct CloudWatch API calls in the hot path.

The traffic controller (Phase 3) is a Lambda function `ab-gateway-traffic-controller` invoked on a 15-minute EventBridge schedule. It reads from both the `ABGateway` EMF namespace and the `AWS/SageMaker` native namespace, then scores each variant using `score = confidence × (1/normalised_latency) × (1 − error_rate)`. A 0.30 dampening factor prevents oscillation between rebalancing cycles. A 5% floor is enforced per variant to keep all three receiving traffic. If a variant's error rate exceeds 10%, the controller kills its traffic weight to 0. Updated weights are written back to DynamoDB and applied to the endpoint via `UpdateEndpointWeightsAndCapacities`.

The CI/CD pipeline (Phase 4) is a self-mutating CDK Pipeline (`ab-gateway-infra-pipeline`) triggered by git push to main. Model deployment is triggered separately: Model Registry approval events flow through EventBridge (`ab-gateway-model-approved`) to a canary Lambda (30-second delay, then starts the state machine) to a Step Functions Standard Workflow. The workflow runs six states: start validation (CodeBuild job invoking the endpoint via `TargetVariant` pinning on the 200-example golden set), wait for CodeBuild to complete, poll result, shift 10% traffic weight to the new variant, wait 15 minutes, then health check — promoting on pass, rolling back on failure.

Step Functions Standard Workflow was chosen over Express because Lambda's maximum timeout is 15 minutes, leaving zero margin for a 15-minute baking wait plus validation time. Express Workflows cap at 5 minutes — also insufficient. `UpdateEndpointWeightsAndCapacities` was chosen over `UpdateEndpoint` because `UpdateEndpoint` with a new EndpointConfig provisions a second fleet via blue/green swap, takes roughly 10 minutes, and doubles instance cost during the transition. `UpdateEndpointWeightsAndCapacities` shifts traffic weights on the existing fleet in seconds with no additional cost and preserves the traffic controller's accumulated weight rebalancing.

---

## 8. What Would Be Different in Production

- The manual approval step in the canary pipeline is currently bypassed with `SKIP_MANUAL_APPROVAL=true`. A production deployment would use Step Functions `waitForTaskToken`: the state machine emits a task token, pauses, and only resumes when an approval Lambda — sitting behind an API Gateway endpoint — receives a human approval and calls `SendTaskSuccess` with the token.
- The ~1,200ms fixed infrastructure overhead makes the 1.55x INT8 and 2.08x DistilBERT compute speedups invisible at the client level. The production fix is a VPC endpoint for SageMaker (removes the internet round trip and API Gateway) or a move to SageMaker serverless inference, which eliminates the API Gateway layer and its associated latency.
- `register_models.py` registers models directly with `ModelApprovalStatus=Approved`, bypassing the canary pipeline entirely. This script was written for the initial deployment and should not be used for ongoing model updates — any subsequent model registration should go through the Model Registry with `PendingManualApproval` status and flow through the EventBridge-triggered canary workflow.
- All resources currently live in a single AWS account and region. A production deployment would use separate accounts for staging and production with CDK cross-account trust relationships, so that a broken canary execution cannot affect the production endpoint.
- The traffic controller runs every 15 minutes with no safeguard against low invocation volume. A `ControllerHeartbeatMissed` alarm is already deployed, but production would add a circuit breaker: if invocation volume drops below a minimum threshold per rebalancing cycle, weight updates are paused to prevent the controller from making statistically meaningless routing decisions from sparse data.
- The `CheckCanaryHealth` state defaults to `canary_healthy=true` when CloudWatch has no data for the canary variant during the baking window — confirmed in execution `canary-VariantB-BERT-INT8-1774605948`, where the health check returned `NO_DATA — assumed healthy (insufficient traffic during baking)` and promoted without measuring a single live request; production requires a minimum invocation count during the bake window before the health check is permitted to return a verdict.

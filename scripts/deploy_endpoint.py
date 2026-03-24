"""
Deploy all three model variants as a single multi-variant SageMaker endpoint.

Variants A & C: model weights stay in their raw training output S3 paths.
                Inference code (inference.py) is packaged locally into a tiny
                tar.gz (<5 KB) and uploaded to S3 once via SAGEMAKER_SUBMIT_DIRECTORY.
Variant B:      bert-int8.tar.gz already contains code/ — no separate upload needed.

Writes the endpoint name to .endpoint_name for use by downstream scripts.
"""
import argparse
import sys
import os
import json
import tarfile
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import boto3

sm_client  = boto3.client("sagemaker", region_name=config.AWS_REGION)
s3_client  = boto3.client("s3",        region_name=config.AWS_REGION)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENDPOINT_NAME_FILE = os.path.join(REPO_ROOT, ".endpoint_name")


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy multi-variant SageMaker endpoint")
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without calling the API")
    return parser.parse_args()


def _make_code_tar(source_dir: str) -> str:
    """Pack source_dir into a temp tar.gz, return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    tmp.close()
    with tarfile.open(tmp.name, "w:gz") as tar:
        for fname in os.listdir(source_dir):
            tar.add(os.path.join(source_dir, fname), arcname=fname)
    return tmp.name


def _upload_code(local_tar: str, s3_uri: str):
    """Upload local_tar to s3_uri (s3://bucket/key)."""
    parts   = s3_uri.replace("s3://", "").split("/", 1)
    bucket  = parts[0]
    key     = parts[1]
    print(f"  Uploading inference code → {s3_uri}")
    s3_client.upload_file(local_tar, bucket, key)


def upload_inference_code():
    """Build and upload code tarballs for FP32 and ONNX inference scripts."""
    fp32_src  = os.path.join(REPO_ROOT, "models", "inference_fp32")
    onnx_src  = os.path.join(REPO_ROOT, "models", "inference_onnx")

    fp32_tar = _make_code_tar(fp32_src)
    onnx_tar = _make_code_tar(onnx_src)

    _upload_code(fp32_tar, config.S3_CODE_FP32)
    _upload_code(onnx_tar, config.S3_CODE_ONNX)

    os.unlink(fp32_tar)
    os.unlink(onnx_tar)


def main():
    args = parse_args()

    timestamp     = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    endpoint_name = args.endpoint_name or f"{config.ENDPOINT_NAME_PREFIX}-{timestamp}"
    model_name_map = {
        v: f"{v.replace('/', '-')}-{timestamp}" for v in config.ALL_VARIANTS
    }
    endpoint_config_name = f"{endpoint_name}-config"

    # Per-variant model config: (model_data_s3, code_s3_or_None, inference_program)
    variant_cfg = {
        config.VARIANT_A: (config.S3_BERT_FP32,  config.S3_CODE_FP32,  "inference.py"),
        config.VARIANT_B: (config.S3_BERT_INT8,   None,                 "inference.py"),
        config.VARIANT_C: (config.S3_DISTILBERT,  config.S3_CODE_FP32,  "inference.py"),
    }

    production_variants = [
        {
            "VariantName":          v,
            "ModelName":            model_name_map[v],
            "InitialInstanceCount": 1,
            "InstanceType":         config.INSTANCE_TYPE_INFER,
            "InitialVariantWeight": config.INITIAL_WEIGHTS[v],
        }
        for v in config.ALL_VARIANTS
    ]

    if args.dry_run:
        print("=== DRY RUN — no API calls will be made ===\n")
        print(f"Endpoint name  : {endpoint_name}")
        print(f"Endpoint config: {endpoint_config_name}\n")
        for v in config.ALL_VARIANTS:
            model_data, code_s3, program = variant_cfg[v]
            print(f"  {model_name_map[v]}")
            print(f"    weights : {model_data}")
            print(f"    code    : {code_s3 or '(bundled in model.tar.gz)'}")
        print("\nProduction variants:")
        total = sum(config.INITIAL_WEIGHTS.values())
        for pv in production_variants:
            w = pv["InitialVariantWeight"]
            print(f"  {pv['VariantName']}: weight={w} ({int(w/total*100)}%)")
        return

    # ── 1. Upload inference code tarballs (tiny — <5 KB each) ────────────────
    print("Uploading inference code to S3...")
    upload_inference_code()

    # ── 2. Create SageMaker Model objects ────────────────────────────────────
    print("\nCreating SageMaker Model objects...")
    for v in config.ALL_VARIANTS:
        model_data, code_s3, program = variant_cfg[v]
        model_name = model_name_map[v]
        env = {
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_REGION":              config.AWS_REGION,
            "SAGEMAKER_PROGRAM":             program,
        }
        if code_s3:
            env["SAGEMAKER_SUBMIT_DIRECTORY"] = code_s3

        print(f"  Creating model: {model_name}")
        sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image":        config.HF_DLC_IMAGE,
                "ModelDataUrl": model_data,
                "Environment":  env,
            },
            ExecutionRoleArn=config.SAGEMAKER_ROLE,
        )

    # ── 3. Create endpoint config ─────────────────────────────────────────────
    print(f"\nCreating endpoint config: {endpoint_config_name}")
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=production_variants,
    )

    # ── 4. Create endpoint ────────────────────────────────────────────────────
    print(f"\nCreating endpoint: {endpoint_name}")
    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )

    # ── 5. Wait for InService ─────────────────────────────────────────────────
    print("Waiting for endpoint to be InService (this takes ~10 minutes)...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 60},
    )

    # ── 6. Save endpoint name ─────────────────────────────────────────────────
    with open(ENDPOINT_NAME_FILE, "w") as f:
        f.write(endpoint_name)

    print("\n" + "=" * 60)
    print("ENDPOINT DEPLOYED SUCCESSFULLY")
    print("=" * 60)
    print(f"Endpoint name : {endpoint_name}")
    print(f"Saved to      : {ENDPOINT_NAME_FILE}")
    total = sum(config.INITIAL_WEIGHTS.values())
    for v in config.ALL_VARIANTS:
        w = config.INITIAL_WEIGHTS[v]
        print(f"  {v}: {int(w/total*100)}%")
    print("=" * 60)
    print(f"\nRun validation:\n  python scripts/validate_endpoint.py")


if __name__ == "__main__":
    main()

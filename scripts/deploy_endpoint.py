"""
Deploy all three model variants as a single multi-variant SageMaker endpoint.

All three variants bundle code/inference.py inside their model.tar.gz so the
HuggingFace DLC finds it automatically at /opt/ml/model/code/inference.py.
No SAGEMAKER_SUBMIT_DIRECTORY or SAGEMAKER_PROGRAM env vars are used — they
are unreliable with raw boto3 create_model() calls.

Variants A & C: original model weights are downloaded from S3, inference code
                is injected under code/, and the result is re-uploaded to a new
                S3 path before create_model() is called.
Variant B:      bert-int8.tar.gz already has code/ bundled correctly.

Writes the endpoint name to .endpoint_name for use by downstream scripts.
"""
import argparse
import sys
import os
import tarfile
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import boto3

sm_client = boto3.client("sagemaker", region_name=config.AWS_REGION)
s3_client = boto3.client("s3",        region_name=config.AWS_REGION)

REPO_ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDPOINT_NAME_FILE = os.path.join(REPO_ROOT, ".endpoint_name")
_SKIP              = {"__pycache__", ".pyc", ".DS_Store"}


def _should_skip(name: str) -> bool:
    return any(s in name for s in _SKIP)


def _parse_s3(uri: str):
    parts = uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]


def _repackage_model_with_code(model_s3_uri: str, code_src_dir: str, output_s3_uri: str) -> None:
    """
    Download model.tar.gz from S3, inject code/inference.py + code/requirements.txt,
    and re-upload to output_s3_uri.

    The HuggingFace DLC looks for inference.py at /opt/ml/model/code/inference.py.
    Bundling it inside the model archive is the only reliable way to ensure it is
    found — SAGEMAKER_SUBMIT_DIRECTORY is ignored by raw boto3 create_model() calls.
    """
    src_bucket, src_key = _parse_s3(model_s3_uri)
    dst_bucket, dst_key = _parse_s3(output_s3_uri)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download original model archive
        orig_tar = os.path.join(tmpdir, "orig.tar.gz")
        print(f"    Downloading {model_s3_uri} ...")
        s3_client.download_file(src_bucket, src_key, orig_tar)

        # Unpack model artifacts
        model_dir = os.path.join(tmpdir, "model")
        os.makedirs(model_dir)
        with tarfile.open(orig_tar, "r:gz") as tf:
            tf.extractall(model_dir)

        # Repack: model artifacts at root + inference code under code/
        new_tar = os.path.join(tmpdir, "model_with_code.tar.gz")
        with tarfile.open(new_tar, "w:gz") as tar:
            for fname in os.listdir(model_dir):
                if not _should_skip(fname):
                    tar.add(os.path.join(model_dir, fname), arcname=fname)
            for fname in os.listdir(code_src_dir):
                if not _should_skip(fname):
                    tar.add(
                        os.path.join(code_src_dir, fname),
                        arcname=os.path.join("code", fname),
                    )

        # Verify before uploading
        _verify_tarball(new_tar)

        print(f"    Uploading bundled archive → {output_s3_uri}")
        s3_client.upload_file(new_tar, dst_bucket, dst_key)


def _verify_tarball(tar_path: str) -> None:
    """Abort early if code/inference.py is missing from the archive."""
    with tarfile.open(tar_path, "r:gz") as tf:
        names = tf.getnames()

    assert "code/inference.py" in names, (
        f"code/inference.py not found in {tar_path}.\n"
        f"Contents: {sorted(names)}"
    )
    bad = [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
    assert not bad, f"Tarball contains cache files that should be excluded: {bad}"
    print(f"    Verified: code/inference.py present, no __pycache__ files.")


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy multi-variant SageMaker endpoint")
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument(
        "--instance-type",
        default=config.INSTANCE_TYPE_INFER,
        help=f"Instance type for all variants (default: {config.INSTANCE_TYPE_INFER})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without calling the API",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp            = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    endpoint_name        = args.endpoint_name or f"{config.ENDPOINT_NAME_PREFIX}-{timestamp}"
    endpoint_config_name = f"{endpoint_name}-config"
    model_name_map       = {v: f"{v}-{timestamp}" for v in config.ALL_VARIANTS}
    instance_type        = args.instance_type

    # Bundled S3 paths used for create_model()
    variant_model_data = {
        config.VARIANT_A: config.S3_BERT_FP32_BUNDLED,
        config.VARIANT_B: config.S3_BERT_INT8,           # already bundled
        config.VARIANT_C: config.S3_DISTILBERT_BUNDLED,
    }

    production_variants = [
        {
            "VariantName":          v,
            "ModelName":            model_name_map[v],
            "InitialInstanceCount": 1,
            "InstanceType":         instance_type,
            "InitialVariantWeight": config.INITIAL_WEIGHTS[v],
        }
        for v in config.ALL_VARIANTS
    ]

    if args.dry_run:
        print("=== DRY RUN — no API calls will be made ===\n")
        print(f"Endpoint name  : {endpoint_name}")
        print(f"Endpoint config: {endpoint_config_name}")
        print(f"Instance type  : {instance_type}\n")
        for v in config.ALL_VARIANTS:
            print(f"  {model_name_map[v]}")
            print(f"    model data: {variant_model_data[v]}")
        print("\nProduction variants:")
        total = sum(config.INITIAL_WEIGHTS.values())
        for pv in production_variants:
            w = pv["InitialVariantWeight"]
            print(f"  {pv['VariantName']}: {instance_type}, weight={w} ({int(w/total*100)}%)")
        return

    # ── 0. Delete existing endpoint from .endpoint_name ──────────────────────
    if os.path.exists(ENDPOINT_NAME_FILE):
        existing = open(ENDPOINT_NAME_FILE).read().strip()
        if existing:
            print(f"Deleting existing endpoint: {existing}")
            try:
                sm_client.delete_endpoint(EndpointName=existing)
                print("  Waiting for deletion...")
                sm_client.get_waiter("endpoint_deleted").wait(EndpointName=existing)
                print("  Deleted.")
            except sm_client.exceptions.ClientError:
                print("  Endpoint not found — skipping deletion.")

    # ── 1. Build and upload bundled model archives for Variants A and C ───────
    print("\nBuilding bundled model archives (weights + code/inference.py)...")
    fp32_code_dir = os.path.join(REPO_ROOT, "models", "inference_fp32")

    print(f"  VariantA-BERT-FP32:")
    _repackage_model_with_code(config.S3_BERT_FP32, fp32_code_dir, config.S3_BERT_FP32_BUNDLED)

    print(f"  VariantC-DistilBERT:")
    _repackage_model_with_code(config.S3_DISTILBERT, fp32_code_dir, config.S3_DISTILBERT_BUNDLED)

    print("  VariantB-BERT-INT8: code already bundled in bert-int8.tar.gz — skipping.")

    # ── 2. Create SageMaker Model objects ─────────────────────────────────────
    print("\nCreating SageMaker Model objects...")
    for v in config.ALL_VARIANTS:
        model_name = model_name_map[v]

        # No SAGEMAKER_SUBMIT_DIRECTORY or SAGEMAKER_PROGRAM — the DLC finds
        # code/inference.py inside the model archive automatically.
        env = {
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_REGION":              config.AWS_REGION,
        }

        try:
            sm_client.delete_model(ModelName=model_name)
            print(f"  Deleted existing model: {model_name}")
        except sm_client.exceptions.ClientError:
            pass

        print(f"  Creating model: {model_name}")
        sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image":        config.HF_DLC_IMAGE,
                "ModelDataUrl": variant_model_data[v],
                "Environment":  env,
            },
            ExecutionRoleArn=config.SAGEMAKER_ROLE,
        )

    # ── 3. Create endpoint config ─────────────────────────────────────────────
    print(f"\nCreating endpoint config: {endpoint_config_name}")
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    except sm_client.exceptions.ClientError:
        pass
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
    print("Waiting for endpoint to be InService (~10 minutes)...")
    sm_client.get_waiter("endpoint_in_service").wait(
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
    print("\nVerify container startup logs — you should see:")
    print('  "Using user script at /opt/ml/model/code/inference.py"')
    print("NOT:")
    print('  "No inference script implementation was found at `inference`"')
    print("\nRun validation:\n  python scripts/validate_endpoint.py")


if __name__ == "__main__":
    main()

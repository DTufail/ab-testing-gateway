"""
Register all three model variants in SageMaker Model Registry.
Creates ModelPackageGroups if they don't exist, then registers each variant.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import boto3
from botocore.exceptions import ClientError

sm_client = boto3.client("sagemaker", region_name=config.AWS_REGION)

# S3 paths for each variant — bundled archives (model weights + code/inference.py)
VARIANT_S3_MAP = {
    config.VARIANT_A: f"s3://{config.S3_BUCKET}/models/bert-fp32/model.tar.gz",
    config.VARIANT_B: f"s3://{config.S3_BUCKET}/models/bert-int8/model.tar.gz",
    config.VARIANT_C: f"s3://{config.S3_BUCKET}/models/distilbert-fp32/model.tar.gz",
}

registered_arns = {}


def ensure_model_package_group(group_name):
    """Create the ModelPackageGroup if it doesn't already exist."""
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=f"BANKING77 intent classifier — {group_name}",
        )
        print(f"  Created ModelPackageGroup: {group_name}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        if code in ("ConflictException", "ValidationException") and "already exists" in msg:
            print(f"  ModelPackageGroup already exists: {group_name} (skipping)")
        else:
            raise


def register_model(variant_name, group_name, s3_uri):
    """Register a model package and return its ARN."""
    print(f"\nRegistering {variant_name}...")
    print(f"  Group: {group_name}")
    print(f"  S3 URI: {s3_uri}")

    ensure_model_package_group(group_name)

    response = sm_client.create_model_package(
        ModelPackageGroupName=group_name,
        ModelPackageDescription=f"{variant_name} fine-tuned on BANKING77",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": config.HF_DLC_IMAGE,
                    "ModelDataUrl": s3_uri,
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="Approved",
    )

    arn = response["ModelPackageArn"]
    print(f"  Registered ARN: {arn}")
    return arn


def main():
    print("=== SageMaker Model Registry — Registering Phase 1 Variants ===\n")

    for variant_name in config.ALL_VARIANTS:
        group_name = config.MODEL_GROUPS[variant_name]
        s3_uri = VARIANT_S3_MAP[variant_name]
        arn = register_model(variant_name, group_name, s3_uri)
        registered_arns[variant_name] = arn

    print("\n" + "=" * 60)
    print("Registration Summary:")
    print("=" * 60)
    print(f"{'Variant':<30} {'Group':<30} {'ARN'}")
    print("-" * 100)
    for variant_name in config.ALL_VARIANTS:
        group_name = config.MODEL_GROUPS[variant_name]
        arn = registered_arns[variant_name]
        print(f"{variant_name:<30} {group_name:<30} {arn}")
    print("=" * 60)


if __name__ == "__main__":
    main()

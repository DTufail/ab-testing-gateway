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

# S3 paths for each variant
VARIANT_S3_MAP = {
    config.VARIANT_A: config.S3_BERT_FP32,
    config.VARIANT_B: config.S3_BERT_INT8,
    config.VARIANT_C: config.S3_DISTILBERT,
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
        if e.response["Error"]["Code"] == "ConflictException":
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

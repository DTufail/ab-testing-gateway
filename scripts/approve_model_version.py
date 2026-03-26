"""
scripts/approve_model_version.py — Approve the latest PendingManualApproval
model version in a SageMaker Model Registry group.

Usage:
    python scripts/approve_model_version.py --group VariantB-BERT-INT8-ModelGroup

This triggers the EventBridge rule ab-gateway-model-approved, which invokes
the canary Lambda and starts the Step Functions state machine.
"""
import argparse
import boto3
import sys

REGION = "us-east-1"
sm = boto3.client("sagemaker", region_name=REGION)


def get_latest_pending(group_name: str) -> str:
    """Return the ARN of the latest PendingManualApproval package in the group."""
    paginator = sm.get_paginator("list_model_packages")
    for page in paginator.paginate(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus="PendingManualApproval",
        SortBy="CreationTime",
        SortOrder="Descending",
    ):
        packages = page.get("ModelPackageSummaryList", [])
        if packages:
            return packages[0]["ModelPackageArn"]
    return None


def approve(arn: str) -> None:
    sm.update_model_package(
        ModelPackageArn=arn,
        ModelApprovalStatus="Approved",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, help="Model package group name")
    args = parser.parse_args()

    print(f"Looking for PendingManualApproval versions in: {args.group}")
    arn = get_latest_pending(args.group)

    if not arn:
        print(f"ERROR: No PendingManualApproval model versions found in {args.group!r}")
        print("Register and submit a model version first, or check the group name.")
        sys.exit(1)

    print(f"Found: {arn}")
    approve(arn)
    print(f"Approved. EventBridge rule will now trigger the canary deployment.")
    print(f"Watch the state machine at:")
    print(f"  https://{REGION}.console.aws.amazon.com/states/home?region={REGION}#/statemachines")


if __name__ == "__main__":
    main()

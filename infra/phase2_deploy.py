"""
Phase 2 deployment: DynamoDB + Lambda + API Gateway + CloudWatch warming rule.

Run order:
  1. Create DynamoDB table
  2. Create IAM role for Lambda
  3. Package and create Lambda function
  4. Create API Gateway REST API (POST /predict → Lambda proxy)
  5. Add Lambda invocation permission for API Gateway
  6. Create CloudWatch Events warming rule
  7. Write benchmarks/phase2_outputs.json

Idempotent: running twice skips resources that already exist.
"""
import json
import os
import sys
import time
import zipfile
import subprocess
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
LAMBDA_DIR   = REPO_ROOT / "lambda" / "router"
BENCHMARKS   = REPO_ROOT / "benchmarks"
ENDPOINT_FILE = REPO_ROOT / ".endpoint_name"

sys.path.insert(0, str(REPO_ROOT))
import config as cfg

# ---------------------------------------------------------------------------
# AWS clients
# ---------------------------------------------------------------------------
REGION     = cfg.AWS_REGION
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

iam_client    = boto3.client("iam",          region_name=REGION)
lambda_client = boto3.client("lambda",       region_name=REGION)
dynamodb      = boto3.client("dynamodb",     region_name=REGION)
apigw         = boto3.client("apigateway",   region_name=REGION)
events_client = boto3.client("events",       region_name=REGION)
logs_client   = boto3.client("logs",         region_name=REGION)

TABLE_NAME       = "ab-gateway-routing-config"
FUNCTION_NAME    = "ab-gateway-router"
ROLE_NAME        = "ab-gateway-router-role"
API_NAME         = "ab-gateway-api"
WARMER_RULE_NAME = "ab-gateway-router-warmer"


# ---------------------------------------------------------------------------
# Step 1 — DynamoDB table
# ---------------------------------------------------------------------------
def create_dynamodb_table() -> str:
    print("[1/7] Creating DynamoDB table...")
    try:
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "config_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "config_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        # Wait for table to become active
        waiter = boto3.client("dynamodb", region_name=REGION).get_waiter("table_exists")
        waiter.wait(TableName=TABLE_NAME)
        print(f"  Created table: {TABLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ResourceInUseException", "ResourceNotFoundException"):
            print(f"  Table already exists: {TABLE_NAME}")
        else:
            raise

    table_arn = f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/{TABLE_NAME}"
    return table_arn


# ---------------------------------------------------------------------------
# Step 2 — IAM role
# ---------------------------------------------------------------------------
def create_lambda_role(table_arn: str, endpoint_name: str) -> str:
    print("[2/7] Creating IAM role for Lambda...")

    trust_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })

    try:
        role = iam_client.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=trust_policy,
            Description="IAM role for ab-gateway-router Lambda",
        )
        role_arn = role["Role"]["Arn"]
        print(f"  Created role: {ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
            print(f"  Role already exists: {ROLE_NAME}")
        else:
            raise

    # Attach managed policy for basic Lambda execution (CloudWatch Logs)
    try:
        iam_client.attach_role_policy(
            RoleName=ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise

    endpoint_arn = (
        f"arn:aws:sagemaker:{REGION}:{ACCOUNT_ID}:endpoint/{endpoint_name}"
    )

    inline_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["dynamodb:GetItem", "dynamodb:PutItem"],
                "Resource": table_arn,
            },
            {
                "Effect": "Allow",
                "Action": "sagemaker:InvokeEndpoint",
                "Resource": endpoint_arn,
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": "*",
            },
        ],
    })

    iam_client.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="ab-gateway-router-inline",
        PolicyDocument=inline_policy,
    )
    print(f"  Inline policy attached to {ROLE_NAME}")

    # Allow time for IAM propagation
    time.sleep(10)
    return role_arn


# ---------------------------------------------------------------------------
# Step 3 — Package and create Lambda
# ---------------------------------------------------------------------------
def _build_lambda_zip() -> bytes:
    """Install requirements into a temp dir and zip with handler code."""
    tmp_dir = tempfile.mkdtemp(prefix="ab-gateway-lambda-")
    try:
        # Install requirements
        req_file = LAMBDA_DIR / "requirements.txt"
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(req_file),
            "-t", tmp_dir,
            "--quiet",
        ])

        # Copy Lambda source files
        for py_file in LAMBDA_DIR.glob("*.py"):
            shutil.copy(py_file, tmp_dir)

        # Create zip
        zip_buffer_path = tempfile.mktemp(suffix=".zip")
        with zipfile.ZipFile(zip_buffer_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmp_dir):
                for fname in files:
                    full_path = os.path.join(root, fname)
                    arcname = os.path.relpath(full_path, tmp_dir)
                    zf.write(full_path, arcname)

        with open(zip_buffer_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def create_lambda_function(role_arn: str, endpoint_name: str) -> str:
    print("[3/7] Packaging and creating Lambda function...")
    zip_bytes = _build_lambda_zip()
    print(f"  Package size: {len(zip_bytes) / 1024:.1f} KB")

    env_vars = {
        "SAGEMAKER_ENDPOINT_NAME": endpoint_name,
        "ROUTING_CONFIG_TABLE":    TABLE_NAME,
        "AWS_EMF_NAMESPACE":       "ABGateway",
    }

    try:
        resp = lambda_client.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime="python3.12",
            Handler="handler.lambda_handler",
            Role=role_arn,
            Code={"ZipFile": zip_bytes},
            Timeout=30,
            MemorySize=512,
            Environment={"Variables": env_vars},
        )
        lambda_arn = resp["FunctionArn"]
        print(f"  Created Lambda: {FUNCTION_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            # Function exists — update code and config
            lambda_client.update_function_code(
                FunctionName=FUNCTION_NAME,
                ZipFile=zip_bytes,
            )
            # Wait for code update to finish before changing config
            lambda_client.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
            lambda_client.update_function_configuration(
                FunctionName=FUNCTION_NAME,
                Role=role_arn,
                Timeout=30,
                MemorySize=512,
                Environment={"Variables": env_vars},
            )
            lambda_arn = lambda_client.get_function(
                FunctionName=FUNCTION_NAME
            )["Configuration"]["FunctionArn"]
            print(f"  Updated existing Lambda: {FUNCTION_NAME}")
        else:
            raise

    # Wait for function to be active
    waiter = lambda_client.get_waiter("function_active_v2")
    waiter.wait(FunctionName=FUNCTION_NAME)
    return lambda_arn


# ---------------------------------------------------------------------------
# Step 4 — API Gateway REST API
# ---------------------------------------------------------------------------
def create_api_gateway(lambda_arn: str) -> str:
    print("[4/7] Creating API Gateway REST API...")

    # Check for existing API
    existing_apis = apigw.get_rest_apis(limit=500)["items"]
    api = next((a for a in existing_apis if a["name"] == API_NAME), None)

    if api:
        api_id = api["id"]
        print(f"  API already exists: {API_NAME} ({api_id})")
    else:
        api_id = apigw.create_rest_api(
            name=API_NAME,
            description="A/B Testing Gateway — Phase 2",
            endpointConfiguration={"types": ["REGIONAL"]},
        )["id"]
        print(f"  Created API: {API_NAME} ({api_id})")

    # Get root resource
    resources = apigw.get_resources(restApiId=api_id)["items"]
    root = next(r for r in resources if r["path"] == "/")

    # Create /predict resource if missing
    predict_resource = next(
        (r for r in resources if r.get("pathPart") == "predict"), None
    )
    if not predict_resource:
        predict_resource = apigw.create_resource(
            restApiId=api_id,
            parentId=root["id"],
            pathPart="predict",
        )
        print("  Created /predict resource")

    resource_id = predict_resource["id"]

    # Create POST method if missing
    try:
        apigw.get_method(restApiId=api_id, resourceId=resource_id, httpMethod="POST")
        print("  POST method already exists")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NotFoundException":
            apigw.put_method(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod="POST",
                authorizationType="NONE",
                requestParameters={
                    "method.request.header.X-Target-Variant": False,
                },
            )
            print("  Created POST method")
        else:
            raise

    # Create Lambda proxy integration
    lambda_uri = (
        f"arn:aws:apigateway:{REGION}:lambda:path/2015-03-31"
        f"/functions/{lambda_arn}/invocations"
    )
    try:
        apigw.get_integration(restApiId=api_id, resourceId=resource_id, httpMethod="POST")
        print("  Integration already exists")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NotFoundException":
            apigw.put_integration(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod="POST",
                type="AWS_PROXY",
                integrationHttpMethod="POST",
                uri=lambda_uri,
            )
            print("  Created Lambda proxy integration")
        else:
            raise

    # Deploy to prod stage
    apigw.create_deployment(restApiId=api_id, stageName="prod")
    print("  Deployed to prod stage")

    api_url = f"https://{api_id}.execute-api.{REGION}.amazonaws.com/prod/predict"
    return api_id, api_url


# ---------------------------------------------------------------------------
# Step 5 — Lambda permission for API Gateway
# ---------------------------------------------------------------------------
def add_api_gateway_permission(lambda_arn: str, api_id: str) -> None:
    print("[5/7] Adding Lambda permission for API Gateway...")
    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="allow-api-gateway",
            Action="lambda:InvokeFunction",
            Principal="apigateway.amazonaws.com",
            SourceArn=f"arn:aws:execute-api:{REGION}:{ACCOUNT_ID}:{api_id}/*/*/predict",
        )
        print("  Permission added")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            print("  Permission already exists")
        else:
            raise


# ---------------------------------------------------------------------------
# Step 6 — CloudWatch Events warming rule
# ---------------------------------------------------------------------------
def create_warming_rule(lambda_arn: str) -> None:
    print("[6/7] Creating CloudWatch Events warming rule...")
    try:
        events_client.put_rule(
            Name=WARMER_RULE_NAME,
            ScheduleExpression="rate(5 minutes)",
            State="ENABLED",
        )
        events_client.put_targets(
            Rule=WARMER_RULE_NAME,
            Targets=[{
                "Id": "router-warmer",
                "Arn": lambda_arn,
                "Input": json.dumps({"source": "warming-ping"}),
            }],
        )
        print(f"  Warming rule created: {WARMER_RULE_NAME}")
    except ClientError as e:
        print(f"  Warming rule error (may already exist): {e}")

    # Allow EventBridge to invoke Lambda
    try:
        rule_arn = events_client.describe_rule(Name=WARMER_RULE_NAME)["Arn"]
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="allow-cloudwatch-events",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=rule_arn,
        )
        print("  Events permission added")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            print("  Events permission already exists")
        else:
            raise


# ---------------------------------------------------------------------------
# Step 7 — Write outputs
# ---------------------------------------------------------------------------
def write_outputs(api_url: str, lambda_arn: str, role_arn: str) -> None:
    print("[7/7] Writing benchmarks/phase2_outputs.json...")
    outputs = {
        "api_gateway_url":      api_url,
        "lambda_function_name": FUNCTION_NAME,
        "dynamodb_table":       TABLE_NAME,
        "lambda_role_arn":      role_arn,
        "deployed_at":          datetime.now(timezone.utc).isoformat(),
    }
    out_path = BENCHMARKS / "phase2_outputs.json"
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"  Written: {out_path}")
    print(json.dumps(outputs, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Read endpoint name
    if not ENDPOINT_FILE.exists():
        sys.exit(f"ERROR: {ENDPOINT_FILE} not found. Run Phase 1 first.")
    endpoint_name = ENDPOINT_FILE.read_text().strip()
    print(f"Using SageMaker endpoint: {endpoint_name}\n")

    table_arn  = create_dynamodb_table()
    role_arn   = create_lambda_role(table_arn, endpoint_name)
    lambda_arn = create_lambda_function(role_arn, endpoint_name)
    api_id, api_url = create_api_gateway(lambda_arn)
    add_api_gateway_permission(lambda_arn, api_id)
    create_warming_rule(lambda_arn)
    write_outputs(api_url, lambda_arn, role_arn)

    print("\nPhase 2 deployment complete.")
    print(f"API URL: {api_url}")


if __name__ == "__main__":
    main()

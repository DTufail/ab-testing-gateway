"""
scripts/test_pipeline.py — Phase 4 integration tests.

Tests all deployed infrastructure end-to-end.
Run from repo root: python scripts/test_pipeline.py
Exit 0 = all tests passed. Exit 1 = one or more tests failed.
"""
import boto3
import json
import os
import sys
import traceback

# ── Clients ───────────────────────────────────────────────────────────────────
cf      = boto3.client("cloudformation",    region_name="us-east-1")
lam     = boto3.client("lambda",            region_name="us-east-1")
sfn     = boto3.client("stepfunctions",     region_name="us-east-1")
events  = boto3.client("events",            region_name="us-east-1")
cb      = boto3.client("codebuild",         region_name="us-east-1")
s3      = boto3.client("s3",                region_name="us-east-1")
sm      = boto3.client("sagemaker",         region_name="us-east-1")
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

ENDPOINT_NAME_FILE = "benchmarks/endpoint_name.txt"

# ── Test runner ───────────────────────────────────────────────────────────────
results = []


def run_test(name: str, fn):
    """Run a single test function and record PASS/FAIL."""
    print(f"\nRunning: {name} ...", end=" ", flush=True)
    try:
        fn()
        print("PASS")
        results.append((name, True, None))
    except AssertionError as e:
        print(f"FAIL — {e}")
        results.append((name, False, str(e)))
    except Exception as e:
        print(f"ERROR — {e}")
        results.append((name, False, f"Unexpected error: {traceback.format_exc()}"))


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_endpoint_name() -> str:
    with open(ENDPOINT_NAME_FILE) as f:
        return f.read().strip()


def stack_status(stack_name: str) -> str:
    resp = cf.describe_stacks(StackName=stack_name)
    return resp["Stacks"][0]["StackStatus"]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_cloudformation_stacks():
    """Test 1 — CloudFormation stacks in stable state"""
    stable = {"CREATE_COMPLETE", "UPDATE_COMPLETE"}
    for stack_name in [
        "ABGatewayPipelineStack",
        "ModelDeployInfra-ValidationStack",
        "ModelDeployInfra-CanaryStack",
    ]:
        status = stack_status(stack_name)
        assert status in stable, f"{stack_name} is in status {status!r}, expected one of {stable}"


def test_canary_lambda_config():
    """Test 2 — Canary Lambda config"""
    cfg = lam.get_function_configuration(FunctionName="ab-gateway-canary-deploy")
    assert cfg["Runtime"] == "python3.12", \
        f"Runtime is {cfg['Runtime']!r}, expected 'python3.12'"
    assert cfg["Timeout"] == 30, \
        f"Timeout is {cfg['Timeout']}, expected 30"
    sm_arn = cfg.get("Environment", {}).get("Variables", {}).get("STATE_MACHINE_ARN", "")
    assert sm_arn.startswith("arn:aws:states:"), \
        f"STATE_MACHINE_ARN is {sm_arn!r}, expected to start with 'arn:aws:states:'"


def test_state_machine_active():
    """Test 3 — State machine active"""
    paginator = sfn.get_paginator("list_state_machines")
    found = None
    for page in paginator.paginate():
        for sm_info in page["stateMachines"]:
            if sm_info["name"] == "ab-gateway-canary-workflow":
                found = sm_info
                break
        if found:
            break
    assert found is not None, "State machine 'ab-gateway-canary-workflow' not found"

    # describe_state_machine gives us status and type
    detail = sfn.describe_state_machine(stateMachineArn=found["stateMachineArn"])
    assert detail["status"] == "ACTIVE", \
        f"State machine status is {detail['status']!r}, expected 'ACTIVE'"
    assert detail["type"] == "STANDARD", \
        f"State machine type is {detail['type']!r}, expected 'STANDARD'"


def test_eventbridge_rule_enabled():
    """Test 4 — EventBridge rule enabled and correct"""
    rule = events.describe_rule(Name="ab-gateway-model-approved")
    assert rule["State"] == "ENABLED", \
        f"Rule state is {rule['State']!r}, expected 'ENABLED'"

    targets_resp = events.list_targets_by_rule(Rule="ab-gateway-model-approved")
    targets = targets_resp.get("Targets", [])
    assert len(targets) == 1, \
        f"Expected exactly 1 target, found {len(targets)}"
    target_arn = targets[0].get("Arn", "")
    assert "ab-gateway-canary-deploy" in target_arn, \
        f"Target ARN {target_arn!r} does not contain 'ab-gateway-canary-deploy'"


def test_codebuild_project_real_buildspec():
    """Test 5 — CodeBuild project exists with real buildspec"""
    resp = cb.batch_get_projects(names=["ab-gateway-model-validation"])
    projects = resp.get("projects", [])
    assert len(projects) == 1, \
        "CodeBuild project 'ab-gateway-model-validation' not found"

    project = projects[0]
    compute_type = project["environment"]["computeType"]
    assert compute_type == "BUILD_GENERAL1_SMALL", \
        f"computeType is {compute_type!r}, expected 'BUILD_GENERAL1_SMALL'"

    image = project["environment"]["image"]
    assert "standard:7.0" in image, \
        f"Build image {image!r} does not contain 'standard:7.0'"

    # The source buildspec is stored inline as YAML/JSON in source.buildspec
    # For projects configured with from_object, the buildspec is embedded in the project source
    source = project.get("source", {})
    buildspec_raw = source.get("buildspec", "")
    assert "validate_model.py" in buildspec_raw, \
        "Buildspec does not contain 'validate_model.py' — stub may not have been replaced"


def test_s3_artifact_bucket_exists():
    """Test 6 — S3 artifact bucket exists"""
    try:
        s3.head_bucket(Bucket="ab-gateway-validation-011190986627")
    except Exception as e:
        raise AssertionError(f"Bucket 'ab-gateway-validation-011190986627' not accessible: {e}")


def test_endpoint_healthy_with_variants():
    """Test 7 — Endpoint healthy with correct variants"""
    endpoint_name = get_endpoint_name()
    resp = sm.describe_endpoint(EndpointName=endpoint_name)

    status = resp["EndpointStatus"]
    assert status == "InService", \
        f"Endpoint {endpoint_name!r} status is {status!r}, expected 'InService'"

    variant_names = {v["VariantName"] for v in resp["ProductionVariants"]}
    expected = {"VariantA-BERT-FP32", "VariantB-BERT-INT8", "VariantC-DistilBERT"}
    missing = expected - variant_names
    assert not missing, \
        f"Endpoint is missing variants: {missing}. Found: {variant_names}"


def test_endpoint_invocation_dry_run():
    """Test 8 — Endpoint serves VariantB-BERT-INT8 with correct response format"""
    endpoint_name = get_endpoint_name()

    # Load the first 5 examples from the golden test set locally
    golden_path = "benchmarks/golden_test_set.jsonl"
    examples = []
    with open(golden_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
            if len(examples) >= 5:
                break

    assert len(examples) >= 5, \
        f"Expected at least 5 examples in {golden_path}, found {len(examples)}"

    invoke_errors = []
    for i, example in enumerate(examples):
        try:
            resp = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                TargetVariant="VariantB-BERT-INT8",
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps({"text": example["text"]}),
            )
            result = json.loads(resp["Body"].read().decode("utf-8"))
            assert "predicted_id" in result, \
                f"Example {i}: response missing 'predicted_id' field. Got: {result}"
            assert isinstance(result["predicted_id"], int), \
                f"Example {i}: 'predicted_id' is {type(result['predicted_id']).__name__}, expected int"
            assert "confidence" in result, \
                f"Example {i}: response missing 'confidence' field. Got: {result}"
            assert isinstance(result["confidence"], float), \
                f"Example {i}: 'confidence' is {type(result['confidence']).__name__}, expected float"
        except AssertionError:
            raise
        except Exception as e:
            invoke_errors.append(f"Example {i}: {e}")

    if invoke_errors:
        raise AssertionError(
            f"{len(invoke_errors)}/5 invocations failed:\n" + "\n".join(invoke_errors)
        )


# ── Run all tests ─────────────────────────────────────────────────────────────
run_test("1. CloudFormation stacks in stable state",      test_cloudformation_stacks)
run_test("2. Canary Lambda config",                        test_canary_lambda_config)
run_test("3. State machine active",                        test_state_machine_active)
run_test("4. EventBridge rule enabled and correct",        test_eventbridge_rule_enabled)
run_test("5. CodeBuild project with real buildspec",       test_codebuild_project_real_buildspec)
run_test("6. S3 artifact bucket exists",                   test_s3_artifact_bucket_exists)
run_test("7. Endpoint healthy with correct variants",      test_endpoint_healthy_with_variants)
run_test("8. Endpoint invocation dry run (5 examples)",    test_endpoint_invocation_dry_run)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4 INTEGRATION TEST RESULTS")
print("=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
for name, ok, reason in results:
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}")
    if reason and not ok:
        # Indent multi-line tracebacks
        for line in reason.splitlines():
            print(f"         {line}")
print("=" * 60)
print(f"  {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)

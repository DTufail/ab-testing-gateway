"""
Phase 3 deployment: Traffic Controller Lambda + EventBridge + SNS + Alarms + Dashboard.

Run order:
  1. Read endpoint name
  2. Create IAM role for Traffic Controller Lambda
  3. Package and deploy Traffic Controller Lambda
  4. Create SNS topic + email subscription
  5. Create EventBridge rule (15-minute schedule)
  6. Create CloudWatch Alarms (3 alarms)
  7. Create CloudWatch Dashboard (7 panels)
  8. Write benchmarks/phase3_outputs.json

Idempotent: running twice will not fail. All create calls are wrapped to
handle "already exists" responses gracefully.

Usage:
    ALERT_EMAIL=your@email.com python infra/phase3_deploy.py
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
REPO_ROOT      = Path(__file__).resolve().parent.parent
LAMBDA_DIR     = REPO_ROOT / "lambda" / "traffic_controller"
BENCHMARKS     = REPO_ROOT / "benchmarks"
ENDPOINT_FILE  = REPO_ROOT / ".endpoint_name"            # primary source
ENDPOINT_FILE2 = BENCHMARKS / "endpoint_name.txt"        # secondary / spec reference

sys.path.insert(0, str(REPO_ROOT))
import config as cfg

# ---------------------------------------------------------------------------
# AWS clients
# ---------------------------------------------------------------------------
REGION     = cfg.AWS_REGION
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

iam_client    = boto3.client("iam",         region_name=REGION)
lambda_client = boto3.client("lambda",      region_name=REGION)
events_client = boto3.client("events",      region_name=REGION)
cw_client     = boto3.client("cloudwatch",  region_name=REGION)
sns_client    = boto3.client("sns",         region_name=REGION)

TABLE_NAME         = "ab-gateway-routing-config"
TABLE_ARN          = f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/{TABLE_NAME}"
CONTROLLER_NAME    = "ab-gateway-traffic-controller"
CONTROLLER_ROLE    = "ab-gateway-controller-role"
EVENTBRIDGE_RULE   = "ab-gateway-controller-schedule"
SNS_TOPIC_NAME     = "ab-gateway-alerts"
DASHBOARD_NAME     = "ABGateway-Dashboard"
ALARM_ERROR_RATE   = "ABGateway-HighErrorRate"
ALARM_P95_LATENCY  = "ABGateway-HighP95Latency"
ALARM_HEARTBEAT    = "ABGateway-ControllerHeartbeatMissed"


# ---------------------------------------------------------------------------
# Helper: read endpoint name
# ---------------------------------------------------------------------------
def _read_endpoint_name() -> str:
    # Prefer the file written by Phase 1 pipeline
    for path in (ENDPOINT_FILE, ENDPOINT_FILE2):
        if path.exists():
            name = path.read_text().strip()
            if name:
                # Mirror to benchmarks/endpoint_name.txt for pull_sagemaker script
                ENDPOINT_FILE2.parent.mkdir(parents=True, exist_ok=True)
                ENDPOINT_FILE2.write_text(name)
                return name
    sys.exit(
        "ERROR: endpoint name file not found. "
        "Expected .endpoint_name or benchmarks/endpoint_name.txt. "
        "Run Phase 1 first."
    )


# ---------------------------------------------------------------------------
# Step 1 (read) is done in main() — no AWS call needed
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 2 — IAM role for Traffic Controller Lambda
# ---------------------------------------------------------------------------
def create_controller_role(sns_topic_arn: str) -> str:
    print("[2/8] Creating IAM role for Traffic Controller Lambda...")

    trust_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })

    try:
        role_arn = iam_client.create_role(
            RoleName=CONTROLLER_ROLE,
            AssumeRolePolicyDocument=trust_policy,
            Description="IAM role for ab-gateway-traffic-controller Lambda",
        )["Role"]["Arn"]
        print(f"  Created role: {CONTROLLER_ROLE}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            role_arn = iam_client.get_role(RoleName=CONTROLLER_ROLE)["Role"]["Arn"]
            print(f"  Role already exists: {CONTROLLER_ROLE}")
        else:
            raise

    inline_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "cloudwatch:GetMetricStatistics",
                    "cloudwatch:PutDashboard",
                    "cloudwatch:PutMetricAlarm",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": ["dynamodb:GetItem", "dynamodb:UpdateItem"],
                "Resource": TABLE_ARN,
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
            {
                "Effect": "Allow",
                "Action": "sns:Publish",
                "Resource": sns_topic_arn,
            },
        ],
    })

    iam_client.put_role_policy(
        RoleName=CONTROLLER_ROLE,
        PolicyName="ab-gateway-controller-inline",
        PolicyDocument=inline_policy,
    )
    print(f"  Inline policy attached to {CONTROLLER_ROLE}")

    # Allow IAM propagation before Lambda create
    time.sleep(10)
    return role_arn


# ---------------------------------------------------------------------------
# Step 3 — Package and deploy Traffic Controller Lambda
# ---------------------------------------------------------------------------
def _build_controller_zip() -> bytes:
    tmp_dir = tempfile.mkdtemp(prefix="ab-controller-lambda-")
    try:
        req_file = LAMBDA_DIR / "requirements.txt"
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(req_file),
            "-t", tmp_dir,
            "--quiet",
        ])
        for py_file in LAMBDA_DIR.glob("*.py"):
            shutil.copy(py_file, tmp_dir)

        zip_buffer_path = tempfile.mktemp(suffix=".zip")
        with zipfile.ZipFile(zip_buffer_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmp_dir):
                for fname in files:
                    full_path = os.path.join(root, fname)
                    arcname   = os.path.relpath(full_path, tmp_dir)
                    zf.write(full_path, arcname)

        with open(zip_buffer_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def deploy_controller_lambda(role_arn: str, endpoint_name: str) -> str:
    print("[3/8] Packaging and deploying Traffic Controller Lambda...")
    zip_bytes = _build_controller_zip()
    print(f"  Package size: {len(zip_bytes) / 1024:.1f} KB")

    env_vars = {
        "ROUTING_CONFIG_TABLE":    TABLE_NAME,
        "SAGEMAKER_ENDPOINT_NAME": endpoint_name,
    }

    try:
        resp = lambda_client.create_function(
            FunctionName=CONTROLLER_NAME,
            Runtime="python3.12",
            Handler="handler.lambda_handler",
            Role=role_arn,
            Code={"ZipFile": zip_bytes},
            Timeout=120,       # 18 CloudWatch API calls on cold start; 60s was too tight
            MemorySize=256,    # Pure I/O, no compute
            Environment={"Variables": env_vars},
            Description="A/B Gateway Traffic Controller — Phase 3",
        )
        lambda_arn = resp["FunctionArn"]
        print(f"  Created Lambda: {CONTROLLER_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            lambda_client.update_function_code(
                FunctionName=CONTROLLER_NAME,
                ZipFile=zip_bytes,
            )
            lambda_client.get_waiter("function_updated_v2").wait(
                FunctionName=CONTROLLER_NAME
            )
            lambda_client.update_function_configuration(
                FunctionName=CONTROLLER_NAME,
                Role=role_arn,
                Timeout=120,
                MemorySize=256,
                Environment={"Variables": env_vars},
            )
            lambda_arn = lambda_client.get_function(
                FunctionName=CONTROLLER_NAME
            )["Configuration"]["FunctionArn"]
            print(f"  Updated existing Lambda: {CONTROLLER_NAME}")
        else:
            raise

    lambda_client.get_waiter("function_active_v2").wait(
        FunctionName=CONTROLLER_NAME
    )
    return lambda_arn


# ---------------------------------------------------------------------------
# Step 4 — SNS topic + email subscription
# ---------------------------------------------------------------------------
def create_sns_topic() -> str:
    print("[4/8] Creating SNS topic and email subscription...")

    # create_topic is idempotent — returns existing ARN if same name
    topic_arn = sns_client.create_topic(Name=SNS_TOPIC_NAME)["TopicArn"]
    print(f"  SNS topic: {topic_arn}")

    alert_email = os.environ.get("ALERT_EMAIL", "").strip()
    if not alert_email:
        alert_email = input("  Enter alert email address for SNS subscription: ").strip()
    if not alert_email:
        print("  WARNING: No email provided — skipping SNS email subscription.")
        return topic_arn

    sns_client.subscribe(
        TopicArn=topic_arn,
        Protocol="email",
        Endpoint=alert_email,
    )
    print(f"  Subscription request sent to: {alert_email}")

    return topic_arn


# ---------------------------------------------------------------------------
# Step 5 — EventBridge rule (15-minute schedule)
# ---------------------------------------------------------------------------
def create_eventbridge_rule(controller_lambda_arn: str) -> str:
    print("[5/8] Creating EventBridge rule (15-minute schedule)...")

    resp = events_client.put_rule(
        Name=EVENTBRIDGE_RULE,
        ScheduleExpression="rate(15 minutes)",
        State="ENABLED",
        Description="Triggers Traffic Controller to rebalance routing weights",
    )
    rule_arn = resp["RuleArn"]
    print(f"  Rule ARN: {rule_arn}")

    events_client.put_targets(
        Rule=EVENTBRIDGE_RULE,
        Targets=[{
            "Id": "traffic-controller-target",
            "Arn": controller_lambda_arn,
            "Input": json.dumps({"source": "eventbridge-schedule"}),
        }],
    )
    print(f"  Target set: {CONTROLLER_NAME}")

    try:
        lambda_client.add_permission(
            FunctionName=CONTROLLER_NAME,
            StatementId="allow-eventbridge",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=rule_arn,
        )
        print("  EventBridge → Lambda permission added")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            print("  EventBridge permission already exists")
        else:
            raise

    return rule_arn


# ---------------------------------------------------------------------------
# Step 6 — CloudWatch Alarms
# ---------------------------------------------------------------------------
def create_alarms(sns_topic_arn: str) -> None:
    print("[6/8] Creating CloudWatch Alarms...")

    # Alarm 1 — High Error Rate
    cw_client.put_metric_alarm(
        AlarmName=ALARM_ERROR_RATE,
        AlarmDescription="Any variant exceeding 5% error rate in 15-minute window",
        Namespace="ABGateway",
        MetricName="ErrorCount",
        Dimensions=[],
        Statistic="Sum",
        Period=900,
        EvaluationPeriods=1,
        Threshold=5,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[sns_topic_arn],
    )
    print(f"  Created alarm: {ALARM_ERROR_RATE}")

    # Alarm 2 — High p95 Latency
    cw_client.put_metric_alarm(
        AlarmName=ALARM_P95_LATENCY,
        AlarmDescription="End-to-end p95 latency exceeding 5000ms",
        Namespace="ABGateway",
        MetricName="RequestLatency",
        Dimensions=[],
        ExtendedStatistic="p95",
        Period=900,
        EvaluationPeriods=1,
        Threshold=5000,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[sns_topic_arn],
    )
    print(f"  Created alarm: {ALARM_P95_LATENCY}")

    # Alarm 3 — Traffic Controller Heartbeat (Dead Man's Switch)
    cw_client.put_metric_alarm(
        AlarmName=ALARM_HEARTBEAT,
        AlarmDescription="Traffic Controller failed to run in last 30 minutes",
        Namespace="AWS/Events",
        MetricName="SuccessfulInvocationCount",
        Dimensions=[{"Name": "RuleName", "Value": EVENTBRIDGE_RULE}],
        Statistic="Sum",
        Period=1800,
        EvaluationPeriods=1,
        Threshold=1,
        ComparisonOperator="LessThanThreshold",
        TreatMissingData="breaching",   # missing = dead = alarm
        AlarmActions=[sns_topic_arn],
    )
    print(f"  Created alarm: {ALARM_HEARTBEAT}")


# ---------------------------------------------------------------------------
# Step 7 — CloudWatch Dashboard (7 panels)
# ---------------------------------------------------------------------------
def create_dashboard(endpoint_name: str) -> None:
    print("[7/8] Creating CloudWatch Dashboard...")

    variants = [
        "VariantA-BERT-FP32",
        "VariantB-BERT-INT8",
        "VariantC-DistilBERT",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # ── Panel 1: Traffic Distribution ──────────────────────────────────────
    panel1_metrics = [
        ["ABGateway", "InvocationCount",
         "Variant", v, "Strategy", "weighted_random",
         {"label": v, "color": colors[i], "stat": "Sum", "period": 300}]
        for i, v in enumerate(variants)
    ]
    panel1 = {
        "type": "metric",
        "x": 0, "y": 0, "width": 12, "height": 6,
        "properties": {
            "title":   "Traffic Distribution — Invocations per Variant (5m)",
            "metrics": panel1_metrics,
            "view":    "timeSeries",
            "stacked": True,
            "region":  REGION,
            "period":  300,
            "stat":    "Sum",
        },
    }

    # ── Panel 2: E2E Latency p50/p95 ───────────────────────────────────────
    panel2_metrics = []
    for i, v in enumerate(variants):
        panel2_metrics.append([
            "ABGateway", "RequestLatency",
            "Variant", v, "Strategy", "weighted_random",
            {"label": f"{v} p50", "color": colors[i],
             "stat": "p50", "period": 300},
        ])
        panel2_metrics.append([
            "ABGateway", "RequestLatency",
            "Variant", v, "Strategy", "weighted_random",
            {"label": f"{v} p95", "color": colors[i],
             "stat": "p95", "period": 300},
        ])
    panel2 = {
        "type": "metric",
        "x": 12, "y": 0, "width": 12, "height": 6,
        "properties": {
            "title":   "End-to-End Request Latency p50/p95 (5m)",
            "metrics": panel2_metrics,
            "view":    "timeSeries",
            "stacked": False,
            "region":  REGION,
            "yAxis":   {"left": {"label": "ms"}},
        },
    }

    # ── Panel 3: SageMaker Container Latency — metric math ÷1000 ──────────
    # Raw ModelLatency is in microseconds. Divide by 1000 to show ms.
    # Each variant gets a hidden raw metric (mN) and a visible expression (eN).
    panel3_metrics = []
    for i, v in enumerate(variants):
        m_id = f"m{i + 1}"
        e_id = f"e{i + 1}"
        # Hidden raw metric (µs)
        panel3_metrics.append([
            "AWS/SageMaker", "ModelLatency",
            "EndpointName", endpoint_name,
            "VariantName", v,
            {"id": m_id, "visible": False, "stat": "Average", "period": 60},
        ])
        # Visible expression (ms)
        panel3_metrics.append([
            {"expression": f"{m_id}/1000",
             "label": f"{v} avg (ms)",
             "id": e_id,
             "color": colors[i],
             "period": 60},
        ])
    panel3 = {
        "type": "metric",
        "x": 0, "y": 6, "width": 12, "height": 6,
        "properties": {
            "title":   "SageMaker Container ModelLatency — Pure Inference Time (1m)",
            "metrics": panel3_metrics,
            "view":    "timeSeries",
            "stacked": False,
            "region":  REGION,
            "yAxis":   {"left": {"label": "ms"}},
        },
    }

    # ── Panel 4: Error Count ────────────────────────────────────────────────
    panel4_metrics = [
        ["ABGateway", "ErrorCount",
         "Variant", v, "Strategy", "weighted_random",
         {"label": v, "color": colors[i], "stat": "Sum", "period": 300}]
        for i, v in enumerate(variants)
    ]
    panel4 = {
        "type": "metric",
        "x": 12, "y": 6, "width": 12, "height": 6,
        "properties": {
            "title":   "Error Count per Variant (5m)",
            "metrics": panel4_metrics,
            "view":    "timeSeries",
            "stacked": False,
            "region":  REGION,
            "period":  300,
            "stat":    "Sum",
        },
    }

    # ── Panel 5: Confidence Score ───────────────────────────────────────────
    panel5_metrics = [
        ["ABGateway", "ConfidenceScore",
         "Variant", v,
         {"label": v, "color": colors[i], "stat": "Average", "period": 300}]
        for i, v in enumerate(variants)
    ]
    panel5 = {
        "type": "metric",
        "x": 0, "y": 12, "width": 12, "height": 6,
        "properties": {
            "title":   "Average Confidence Score per Variant (5m)",
            "metrics": panel5_metrics,
            "view":    "timeSeries",
            "stacked": False,
            "region":  REGION,
            "period":  300,
            "stat":    "Average",
            "yAxis":   {"left": {"min": 0, "max": 1}},
        },
    }

    # ── Panel 6: CPU Utilization per Variant ───────────────────────────────
    panel6_metrics = [
        ["/aws/sagemaker/Endpoints", "CPUUtilization",
         "EndpointName", endpoint_name,
         "VariantName", v,
         {"label": v, "color": colors[i], "stat": "Average", "period": 60}]
        for i, v in enumerate(variants)
    ]
    panel6 = {
        "type": "metric",
        "x": 12, "y": 12, "width": 12, "height": 6,
        "properties": {
            "title":   "CPU Utilization per Variant (1m)",
            "metrics": panel6_metrics,
            "view":    "timeSeries",
            "stacked": False,
            "region":  REGION,
            "period":  60,
            "stat":    "Average",
            "yAxis":   {"left": {"label": "%"}},
        },
    }

    # ── Panel 7: Traffic Controller Decision Log ────────────────────────────
    panel7_query = (
        "fields @timestamp, new_weights, cycle_metrics.sample_counts, config_version\n"
        "| filter source = \"traffic-controller\"\n"
        "| sort @timestamp desc\n"
        "| limit 20"
    )
    panel7 = {
        "type": "log",
        "x": 0, "y": 18, "width": 24, "height": 6,
        "properties": {
            "title":         "Traffic Controller — Last 20 Weight Update Decisions",
            "query":         f"SOURCE '/aws/lambda/{CONTROLLER_NAME}' | {panel7_query}",
            "region":        REGION,
            "view":          "table",
            "logGroupNames": [f"/aws/lambda/{CONTROLLER_NAME}"],
        },
    }

    dashboard_body = json.dumps({
        "widgets": [panel1, panel2, panel3, panel4, panel5, panel6, panel7]
    })

    cw_client.put_dashboard(
        DashboardName=DASHBOARD_NAME,
        DashboardBody=dashboard_body,
    )
    print(f"  Dashboard created: {DASHBOARD_NAME}")
    print(f"  URL: https://{REGION}.console.aws.amazon.com/cloudwatch/home"
          f"#dashboards:name={DASHBOARD_NAME}")


# ---------------------------------------------------------------------------
# Step 8 — Write outputs
# ---------------------------------------------------------------------------
def write_outputs(
    controller_lambda_arn: str,
    sns_topic_arn: str,
) -> None:
    print("[8/8] Writing benchmarks/phase3_outputs.json...")
    outputs = {
        "controller_lambda_arn":  controller_lambda_arn,
        "controller_lambda_name": CONTROLLER_NAME,
        "eventbridge_rule_name":  EVENTBRIDGE_RULE,
        "sns_topic_arn":          sns_topic_arn,
        "dashboard_name":         DASHBOARD_NAME,
        "alarms": [ALARM_ERROR_RATE, ALARM_P95_LATENCY, ALARM_HEARTBEAT],
        "deployed_at":            datetime.now(timezone.utc).isoformat(),
    }
    out_path = BENCHMARKS / "phase3_outputs.json"
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"  Written: {out_path}")
    print(json.dumps(outputs, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Phase 3 Deployment ===\n")

    # Step 1: Read endpoint name
    print("[1/8] Reading endpoint name...")
    endpoint_name = _read_endpoint_name()
    print(f"  SageMaker endpoint: {endpoint_name}\n")

    # Steps run in dependency order: SNS first (ARN needed for IAM + alarms)
    sns_topic_arn = create_sns_topic()

    role_arn = create_controller_role(sns_topic_arn)

    controller_lambda_arn = deploy_controller_lambda(role_arn, endpoint_name)

    create_eventbridge_rule(controller_lambda_arn)

    create_alarms(sns_topic_arn)

    create_dashboard(endpoint_name)

    write_outputs(controller_lambda_arn, sns_topic_arn)

    print("\n" + "=" * 60)
    print("Phase 3 deployment complete.")
    print("=" * 60)
    print()
    print("IMPORTANT: Check your email and confirm the SNS subscription")
    print("before alarms will deliver email notifications.")
    print()
    print("Next steps:")
    print("  python scripts/pull_sagemaker_native_latency.py")
    print("  python scripts/test_traffic_controller.py")
    print()
    print(f"  Dashboard: https://{REGION}.console.aws.amazon.com/cloudwatch/"
          f"home#dashboards:name={DASHBOARD_NAME}")


if __name__ == "__main__":
    main()

import os
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_events as events,
    aws_events_targets as targets,
    aws_sns as sns,
    Duration,
    CfnOutput,
)
from constructs import Construct


class CanaryStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx     = self.node.try_get_context
        account = self.account
        region  = self.region

        # Read endpoint name at synth time (developer's machine — file exists here).
        # This avoids reading files at Lambda runtime where paths are unreliable.
        endpoint_name_file = os.path.join(
            os.path.dirname(__file__),   # cdk/stacks/
            "..", "..",                   # repo root
            "benchmarks", "endpoint_name.txt",
        )
        try:
            with open(os.path.normpath(endpoint_name_file)) as f:
                endpoint_name = f.read().strip()
        except FileNotFoundError:
            # Fallback for CI environments where the file may not exist.
            # The Lambda will fail at runtime if this placeholder is not replaced.
            endpoint_name = "ENDPOINT_NAME_NOT_FOUND_SET_MANUALLY"

        # ── SNS topic for deployment notifications ─────────────────────────
        self.notification_topic = sns.Topic(
            self,
            "CanaryNotifications",
            topic_name="ab-gateway-canary-notifications",
            display_name="AB Gateway Canary Deployment Notifications",
        )

        # ── IAM role for Canary Lambda ─────────────────────────────────────
        canary_role = iam.Role(
            self,
            "CanaryLambdaRole",
            role_name="ab-gateway-canary-lambda-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            inline_policies={
                "CanaryDeployPolicy": iam.PolicyDocument(
                    statements=[
                        # SageMaker — read endpoint, update weights
                        iam.PolicyStatement(
                            sid="SageMakerWeightUpdates",
                            actions=[
                                "sagemaker:DescribeEndpoint",
                                "sagemaker:UpdateEndpointWeightsAndCapacities",
                                "sagemaker:DescribeModelPackage",
                            ],
                            resources=["*"],
                        ),
                        # CodeBuild — start and poll validation job
                        iam.PolicyStatement(
                            sid="CodeBuildValidation",
                            actions=[
                                "codebuild:StartBuild",
                                "codebuild:BatchGetBuilds",
                            ],
                            resources=[
                                f"arn:aws:codebuild:{region}:{account}:project/"
                                f"ab-gateway-model-validation"
                            ],
                        ),
                        # DynamoDB — read routing config only (never write weights)
                        iam.PolicyStatement(
                            sid="DynamoDBReadOnly",
                            actions=["dynamodb:GetItem"],
                            resources=[
                                f"arn:aws:dynamodb:{region}:{account}:table/"
                                f"ab-gateway-routing-config"
                            ],
                        ),
                        # CloudWatch — read canary health metrics
                        iam.PolicyStatement(
                            sid="CloudWatchMetrics",
                            actions=["cloudwatch:GetMetricStatistics"],
                            resources=["*"],
                        ),
                        # SNS — publish notifications
                        iam.PolicyStatement(
                            sid="SNSPublish",
                            actions=["sns:Publish"],
                            resources=[self.notification_topic.topic_arn],
                        ),
                        # CloudWatch Logs — Lambda execution logs
                        iam.PolicyStatement(
                            sid="CloudWatchLogs",
                            actions=[
                                "logs:CreateLogGroup",
                                "logs:CreateLogStream",
                                "logs:PutLogEvents",
                            ],
                            resources=[
                                f"arn:aws:logs:{region}:{account}:log-group:"
                                f"/aws/lambda/ab-gateway-canary-deploy*"
                            ],
                        ),
                    ]
                )
            },
        )

        # ── Canary Lambda ──────────────────────────────────────────────────
        # Phase 4a: stub handler only.
        # Phase 4b: replaced with real canary deployment logic via Step Functions.
        self.canary_fn = lambda_.Function(
            self,
            "CanaryDeploy",
            function_name="ab-gateway-canary-deploy",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(
                # Path relative to repo root (where cdk synth runs from)
                os.path.normpath(os.path.join(
                    os.path.dirname(__file__),
                    "..", "lambda", "canary_deploy",
                ))
            ),
            role=canary_role,
            # 14 minutes. Phase 4b will replace sleep-based waiting with
            # Step Functions Express Workflow to avoid the 15-min Lambda limit.
            timeout=Duration.seconds(840),
            memory_size=256,
            environment={
                "ENDPOINT_NAME":           endpoint_name,
                "ROUTING_TABLE":           ctx("routing_table"),
                "EMF_NAMESPACE":           ctx("emf_namespace"),
                "CANARY_WEIGHT":           str(ctx("canary_weight")),
                "CANARY_WAIT_MINUTES":     str(ctx("canary_wait_minutes")),
                "ERROR_RATE_THRESHOLD":    str(ctx("error_rate_threshold")),
                "P95_LATENCY_THRESHOLD":   str(ctx("p95_latency_threshold")),
                "NOTIFICATION_TOPIC_ARN":  self.notification_topic.topic_arn,
                "VALIDATION_PROJECT_NAME": "ab-gateway-model-validation",
                # Set to "true" during testing to skip 15-min baking sleep.
                # Phase 4b replaces this with Step Functions wait state.
                "SKIP_BAKING_PERIOD":      "false",
            },
        )

        # ── EventBridge rule — trigger on Model Registry approval ──────────
        # Event source: aws.sagemaker
        # Detail type: "SageMaker Model Package State Change"
        # Condition: ModelApprovalStatus = "Approved"
        # Scope: only our three Model Package Groups
        model_groups = ctx("model_groups")  # list from cdk.json

        approval_rule = events.Rule(
            self,
            "ModelApprovalRule",
            rule_name="ab-gateway-model-approved",
            description=(
                "Trigger canary deployment when a model package in the AB Gateway "
                "model groups is approved in SageMaker Model Registry"
            ),
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Package State Change"],
                detail={
                    "ModelApprovalStatus": ["Approved"],
                    "ModelPackageGroupName": model_groups,
                },
            ),
        )
        approval_rule.add_target(
            targets.LambdaFunction(
                self.canary_fn,
                retry_attempts=1,   # EventBridge retry on Lambda throttle/error
            )
        )

        # ── Outputs ────────────────────────────────────────────────────────
        CfnOutput(
            self, "CanaryLambdaName",
            value=self.canary_fn.function_name,
            export_name="ABGatewayCanaryLambdaName",
        )
        CfnOutput(
            self, "CanaryLambdaArn",
            value=self.canary_fn.function_arn,
            export_name="ABGatewayCanaryLambdaArn",
        )
        CfnOutput(
            self, "NotificationTopicArn",
            value=self.notification_topic.topic_arn,
            export_name="ABGatewayNotificationTopicArn",
        )
        CfnOutput(
            self, "EventBridgeRuleName",
            value=approval_rule.rule_name,
            export_name="ABGatewayApprovalRuleName",
        )
        CfnOutput(
            self, "EndpointNameUsed",
            value=endpoint_name,
            export_name="ABGatewayEndpointName",
        )

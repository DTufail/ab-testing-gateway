import os
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_events as events,
    aws_events_targets as targets,
    aws_sns as sns,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    Duration,
    CfnOutput,
)
from constructs import Construct

# Lambda sources live at cdk/lambda/<name>/
LAMBDA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "lambda")
)


def _lambda_path(name: str) -> str:
    return os.path.join(LAMBDA_DIR, name)


class CanaryStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx     = self.node.try_get_context
        account = self.account
        region  = self.region

        # ── Read endpoint name at synth time ──────────────────────────────
        endpoint_name_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "endpoint_name.txt")
        )
        try:
            with open(endpoint_name_path) as f:
                endpoint_name = f.read().strip()
        except FileNotFoundError:
            endpoint_name = "ENDPOINT_NAME_NOT_FOUND_SET_MANUALLY"

        # ── SNS topic ─────────────────────────────────────────────────────
        notification_topic = sns.Topic(
            self, "CanaryNotifications",
            topic_name="ab-gateway-canary-notifications",
            display_name="AB Gateway Canary Deployment Notifications",
        )

        # ── Shared Lambda environment ─────────────────────────────────────
        shared_env = {
            "ENDPOINT_NAME":           endpoint_name,
            "ROUTING_TABLE":           ctx("routing_table"),
            "EMF_NAMESPACE":           ctx("emf_namespace"),
            "CANARY_WEIGHT":           str(ctx("canary_weight")),
            "CANARY_WAIT_MINUTES":     str(ctx("canary_wait_minutes")),
            "ERROR_RATE_THRESHOLD":    str(ctx("error_rate_threshold")),
            "P95_LATENCY_THRESHOLD":   str(ctx("p95_latency_threshold")),
            "NOTIFICATION_TOPIC_ARN":  notification_topic.topic_arn,
            "VALIDATION_PROJECT_NAME": "ab-gateway-model-validation",
        }

        # ── Shared IAM policy statements ──────────────────────────────────
        sagemaker_policy = iam.PolicyStatement(
            sid="SageMakerWeights",
            actions=[
                "sagemaker:DescribeEndpoint",
                "sagemaker:UpdateEndpointWeightsAndCapacities",
                "sagemaker:DescribeModelPackage",
            ],
            resources=["*"],
        )
        logs_policy = iam.PolicyStatement(
            sid="CloudWatchLogs",
            actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
            resources=[f"arn:aws:logs:{region}:{account}:log-group:/aws/lambda/ab-gateway-*"],
        )
        sns_policy = iam.PolicyStatement(
            sid="SNSPublish",
            actions=["sns:Publish"],
            resources=[notification_topic.topic_arn],
        )

        def _make_role(role_id: str, extra: list = None) -> iam.Role:
            return iam.Role(
                self, role_id,
                assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
                inline_policies={
                    "Base": iam.PolicyDocument(
                        statements=[logs_policy] + (extra or [])
                    )
                },
            )

        def _make_fn(fn_id: str, source_name: str, role: iam.Role,
                     timeout_secs: int = 30, extra_env: dict = None) -> lambda_.Function:
            return lambda_.Function(
                self, fn_id,
                function_name=f"ab-gateway-{source_name.replace('_', '-')}",
                runtime=lambda_.Runtime.PYTHON_3_12,
                handler="handler.lambda_handler",
                code=lambda_.Code.from_asset(_lambda_path(source_name)),
                role=role,
                timeout=Duration.seconds(timeout_secs),
                memory_size=256,
                environment={**shared_env, **(extra_env or {})},
            )

        # ── Lambda functions ──────────────────────────────────────────────
        start_val_fn = _make_fn("StartValidationFn", "start_validation",
            _make_role("StartValRole", [
                iam.PolicyStatement(
                    sid="CodeBuildStart",
                    actions=["codebuild:StartBuild"],
                    resources=[
                        f"arn:aws:codebuild:{region}:{account}:project/ab-gateway-model-validation"
                    ],
                )
            ]))

        check_val_fn = _make_fn("CheckValidationFn", "check_validation",
            _make_role("CheckValRole", [
                iam.PolicyStatement(
                    sid="CodeBuildPoll",
                    actions=["codebuild:BatchGetBuilds"],
                    resources=["*"],
                )
            ]))

        shift_fn   = _make_fn("ShiftWeightFn",   "shift_weight",   _make_role("ShiftWeightRole", [sagemaker_policy]))
        health_fn  = _make_fn("CheckHealthFn",   "check_health",   _make_role("HealthRole", [
            iam.PolicyStatement(
                sid="CloudWatchMetrics",
                actions=["cloudwatch:GetMetricStatistics"],
                resources=["*"],
            )
        ]))
        adjust_fn  = _make_fn("AdjustWeightsFn", "adjust_weights", _make_role("AdjustRole", [sagemaker_policy]))
        notify_fn  = _make_fn("NotifyFn",        "notify",         _make_role("NotifyRole", [sns_policy]))

        # ── State machine definition ───────────────────────────────────────
        # output_path="$.Payload" unwraps the Lambda response so the return dict
        # becomes the state output directly (avoids $.Payload.foo in downstream states).
        def _invoke(state_id: str, fn: lambda_.Function) -> tasks.LambdaInvoke:
            return tasks.LambdaInvoke(
                self, state_id,
                lambda_function=fn,
                output_path="$.Payload",
            )

        # Terminal states
        succeed = sfn.Succeed(self, "DeploymentSucceeded")
        fail    = sfn.Fail(self, "DeploymentFailed",
                           cause="Canary deployment failed — see execution history")

        # ── Promote path ───────────────────────────────────────────────────
        promote_path = (
            sfn.Pass(self, "InjectPromoteAction", parameters={
                "action":               "promote",
                "model_package_arn.$":  "$.model_package_arn",
                "variant_name.$":       "$.variant_name",
                "model_group_name.$":   "$.model_group_name",
                "triggered_at.$":       "$.triggered_at",
                "pre_canary_weights.$": "$.pre_canary_weights",
                "canary_weights_set.$": "$.canary_weights_set",
            })
            .next(_invoke("PromoteWeights", adjust_fn))
            .next(sfn.Pass(self, "InjectSuccessMsg", parameters={
                "subject":              "[AB Gateway] Canary PROMOTED",
                "message":              "Canary deployment successful. Variant promoted to 100%.",
                "model_package_arn.$":  "$.model_package_arn",
                "variant_name.$":       "$.variant_name",
                "model_group_name.$":   "$.model_group_name",
                "triggered_at.$":       "$.triggered_at",
            }))
            .next(_invoke("NotifySuccess", notify_fn))
            .next(succeed)
        )

        # ── Rollback path ──────────────────────────────────────────────────
        rollback_path = (
            sfn.Pass(self, "InjectRollbackAction", parameters={
                "action":               "rollback",
                "model_package_arn.$":  "$.model_package_arn",
                "variant_name.$":       "$.variant_name",
                "model_group_name.$":   "$.model_group_name",
                "triggered_at.$":       "$.triggered_at",
                "pre_canary_weights.$": "$.pre_canary_weights",
                "canary_weights_set.$": "$.canary_weights_set",
            })
            .next(_invoke("RollbackWeights", adjust_fn))
            .next(sfn.Pass(self, "InjectRollbackMsg", parameters={
                "subject":              "[AB Gateway] Canary ROLLED BACK",
                "message":              "Canary health check failed. Weights restored.",
                "model_package_arn.$":  "$.model_package_arn",
                "variant_name.$":       "$.variant_name",
                "model_group_name.$":   "$.model_group_name",
                "triggered_at.$":       "$.triggered_at",
            }))
            .next(_invoke("NotifyRollback", notify_fn))
            .next(fail)
        )

        # ── Health choice ──────────────────────────────────────────────────
        health_choice = (
            sfn.Choice(self, "CanaryHealthy?")
            .when(sfn.Condition.boolean_equals("$.canary_healthy", True), promote_path)
            .otherwise(rollback_path)
        )

        # ── Validated path (shift → bake → health check → choose) ─────────
        validated_path = (
            sfn.Pass(self, "InjectCanaryStartMsg", parameters={
                "subject":              "[AB Gateway] Canary deployment starting",
                "message":              "Validation passed. Shifting 10% traffic to canary variant.",
                "model_package_arn.$":  "$.model_package_arn",
                "variant_name.$":       "$.variant_name",
                "model_group_name.$":   "$.model_group_name",
                "triggered_at.$":       "$.triggered_at",
            })
            .next(_invoke("NotifyCanaryStarting", notify_fn))
            .next(_invoke("ShiftCanaryWeight", shift_fn))
            .next(sfn.Wait(
                self, "BakingPeriod",
                time=sfn.WaitTime.duration(Duration.minutes(ctx("canary_wait_minutes"))),
            ))
            .next(_invoke("CheckCanaryHealth", health_fn))
            .next(health_choice)
        )

        # ── Validation failed path ─────────────────────────────────────────
        validation_failed_path = (
            sfn.Pass(self, "InjectValidationFailedMsg", parameters={
                "subject":              "[AB Gateway] Validation FAILED — canary aborted",
                "message":              "Model validation failed. Canary deployment aborted.",
                "model_package_arn.$":  "$.model_package_arn",
                "variant_name.$":       "$.variant_name",
                "model_group_name.$":   "$.model_group_name",
                "triggered_at.$":       "$.triggered_at",
            })
            .next(_invoke("NotifyValidationFailed", notify_fn))
            .next(fail)
        )

        # ── Core states for validation polling loop ────────────────────────
        start_validation    = _invoke("StartValidation", start_val_fn)
        wait_for_validation = sfn.Wait(
            self, "WaitForValidation",
            time=sfn.WaitTime.duration(Duration.minutes(3)),
        )
        check_validation = _invoke("CheckValidation", check_val_fn)

        # ── Validation status choice ───────────────────────────────────────
        # IMPORTANT: use sfn.Chain.start(wait_for_validation) for the loop-back
        # branch. This references the existing state without calling .next() on it
        # a second time (which CDK would reject with "State already has a next state").
        # The actual wait_for_validation → check_validation transition is set once,
        # in the definition chain below.
        validation_choice = (
            sfn.Choice(self, "ValidationStatus?")
            .when(
                sfn.Condition.string_equals("$.validation_status", "IN_PROGRESS"),
                sfn.Chain.start(wait_for_validation),   # loop back — no new .next() call
            )
            .when(
                sfn.Condition.string_equals("$.validation_status", "FAILED"),
                validation_failed_path,
            )
            .otherwise(validated_path)
        )

        # ── Full definition ────────────────────────────────────────────────
        # Transitions are set exactly once per state:
        #   start_validation  → wait_for_validation (set here)
        #   wait_for_validation → check_validation  (set here)
        #   check_validation  → validation_choice   (set here)
        #   validation_choice: IN_PROGRESS → wait_for_validation (via Chain.start, no .next())
        #   validation_choice: FAILED      → validation_failed_path
        #   validation_choice: otherwise   → validated_path
        definition = (
            start_validation
            .next(wait_for_validation)
            .next(check_validation)
            .next(validation_choice)
        )

        # ── State machine IAM role ─────────────────────────────────────────
        sm_role = iam.Role(
            self, "StateMachineRole",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com"),
            inline_policies={
                "InvokeLambdas": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=["lambda:InvokeFunction"],
                            resources=[
                                start_val_fn.function_arn,
                                check_val_fn.function_arn,
                                shift_fn.function_arn,
                                health_fn.function_arn,
                                adjust_fn.function_arn,
                                notify_fn.function_arn,
                            ],
                        ),
                        iam.PolicyStatement(
                            actions=[
                                "logs:CreateLogGroup", "logs:CreateLogStream",
                                "logs:PutLogEvents", "logs:DescribeLogGroups",
                                "logs:DescribeLogStreams",
                            ],
                            resources=["*"],
                        ),
                    ]
                )
            },
        )

        # ── State machine ──────────────────────────────────────────────────
        state_machine = sfn.StateMachine(
            self, "CanaryStateMachine",
            state_machine_name="ab-gateway-canary-workflow",
            state_machine_type=sfn.StateMachineType.STANDARD,
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            role=sm_role,
            timeout=Duration.hours(2),
        )

        # ── Canary Lambda (thin orchestrator — just starts state machine) ──
        canary_role = iam.Role(
            self, "CanaryLambdaRole",
            role_name="ab-gateway-canary-lambda-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            inline_policies={
                "StartStateMachine": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=["states:StartExecution"],
                            resources=[state_machine.state_machine_arn],
                        ),
                        iam.PolicyStatement(
                            actions=["logs:CreateLogGroup", "logs:CreateLogStream",
                                     "logs:PutLogEvents"],
                            resources=[
                                f"arn:aws:logs:{region}:{account}:log-group:"
                                f"/aws/lambda/ab-gateway-canary-deploy*"
                            ],
                        ),
                    ]
                )
            },
        )

        canary_fn = lambda_.Function(
            self, "CanaryDeploy",
            function_name="ab-gateway-canary-deploy",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(_lambda_path("canary_deploy")),
            role=canary_role,
            timeout=Duration.seconds(30),   # Just starts state machine — no long work
            memory_size=256,
            environment={
                "STATE_MACHINE_ARN": state_machine.state_machine_arn,
            },
        )

        # ── EventBridge rule ──────────────────────────────────────────────
        model_groups  = ctx("model_groups")
        approval_rule = events.Rule(
            self, "ModelApprovalRule",
            rule_name="ab-gateway-model-approved",
            description=(
                "Trigger canary deployment when a model package is approved "
                "in SageMaker Model Registry"
            ),
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Package State Change"],
                detail={
                    "ModelApprovalStatus":    ["Approved"],
                    "ModelPackageGroupName":  model_groups,
                },
            ),
        )
        approval_rule.add_target(targets.LambdaFunction(canary_fn, retry_attempts=1))

        # ── Outputs ────────────────────────────────────────────────────────
        CfnOutput(self, "StateMachineArn",
                  value=state_machine.state_machine_arn,
                  export_name="ABGatewayStateMachineArn")
        CfnOutput(self, "StateMachineName",
                  value=state_machine.state_machine_name,
                  export_name="ABGatewayStateMachineName")
        CfnOutput(self, "CanaryLambdaName",
                  value=canary_fn.function_name,
                  export_name="ABGatewayCanaryLambdaName")
        CfnOutput(self, "NotificationTopicArn",
                  value=notification_topic.topic_arn,
                  export_name="ABGatewayNotificationTopicArn")
        CfnOutput(self, "EventBridgeRuleName",
                  value=approval_rule.rule_name,
                  export_name="ABGatewayApprovalRuleName")
        CfnOutput(self, "EndpointNameUsed",
                  value=endpoint_name,
                  export_name="ABGatewayEndpointName")

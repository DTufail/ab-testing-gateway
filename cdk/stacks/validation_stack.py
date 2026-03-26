import os
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_codebuild as codebuild,
    aws_iam as iam,
    aws_s3 as s3,
    CfnOutput,
    RemovalPolicy,
)
from constructs import Construct


class ValidationStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

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

        # ── S3 bucket for validation artifacts ────────────────────────────
        # Deterministic name so canary Lambda can reference it without
        # CloudFormation cross-stack imports (which add fragile dependencies).
        self.artifact_bucket = s3.Bucket(
            self,
            "ValidationArtifacts",
            bucket_name=f"ab-gateway-validation-{account}",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            versioned=False,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # ── IAM role for CodeBuild ─────────────────────────────────────────
        # Needs SageMaker InvokeEndpoint + S3 (golden set upload/download) + Logs
        # Does NOT use SageMakerFullAccess — principle of least privilege.
        self.validation_role = iam.Role(
            self,
            "ValidationCodeBuildRole",
            role_name="ab-gateway-validation-codebuild-role",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
            inline_policies={
                "ValidationPolicy": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            sid="SageMakerInvokeEndpoint",
                            actions=["sagemaker:InvokeEndpoint"],
                            resources=[
                                f"arn:aws:sagemaker:{region}:{account}:endpoint/*"
                            ],
                        ),
                        iam.PolicyStatement(
                            sid="S3ArtifactAccess",
                            actions=[
                                "s3:GetObject", "s3:PutObject",
                                "s3:DeleteObject", "s3:ListBucket",
                            ],
                            resources=[
                                self.artifact_bucket.bucket_arn,
                                f"{self.artifact_bucket.bucket_arn}/*",
                            ],
                        ),
                        iam.PolicyStatement(
                            sid="CloudWatchLogs",
                            actions=[
                                "logs:CreateLogGroup",
                                "logs:CreateLogStream",
                                "logs:PutLogEvents",
                            ],
                            resources=[
                                f"arn:aws:logs:{region}:{account}:log-group:"
                                f"/aws/codebuild/ab-gateway-model-validation*"
                            ],
                        ),
                    ]
                )
            },
        )

        # ── CodeBuild project ──────────────────────────────────────────────
        # The buildspec is a stub for Phase 4a.
        # Phase 4c will replace this with the real validate_model.py invocation.
        self.validation_project = codebuild.Project(
            self,
            "GoldenDatasetValidation",
            project_name="ab-gateway-model-validation",
            role=self.validation_role,
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.SMALL,
            ),
            environment_variables={
                "ARTIFACT_BUCKET": codebuild.BuildEnvironmentVariable(
                    value=self.artifact_bucket.bucket_name,
                    type=codebuild.BuildEnvironmentVariableType.PLAINTEXT,
                ),
                # These two are overridden at runtime by the Canary Lambda.
                # Default values here are placeholders only.
                "MODEL_PACKAGE_ARN": codebuild.BuildEnvironmentVariable(
                    value="OVERRIDE_AT_RUNTIME",
                    type=codebuild.BuildEnvironmentVariableType.PLAINTEXT,
                ),
                "VARIANT_NAME": codebuild.BuildEnvironmentVariable(
                    value="OVERRIDE_AT_RUNTIME",
                    type=codebuild.BuildEnvironmentVariableType.PLAINTEXT,
                ),
                "SAGEMAKER_EXECUTION_ROLE_ARN": codebuild.BuildEnvironmentVariable(
                    value=self.validation_role.role_arn,
                    type=codebuild.BuildEnvironmentVariableType.PLAINTEXT,
                ),
                "ENDPOINT_NAME": codebuild.BuildEnvironmentVariable(
                    value=endpoint_name,
                    type=codebuild.BuildEnvironmentVariableType.PLAINTEXT,
                ),
                "GOLDEN_S3_KEY": codebuild.BuildEnvironmentVariable(
                    value="validation/golden_test_set.jsonl",
                    type=codebuild.BuildEnvironmentVariableType.PLAINTEXT,
                ),
                "ACCURACY_THRESHOLD": codebuild.BuildEnvironmentVariable(
                    value="0.80",
                    type=codebuild.BuildEnvironmentVariableType.PLAINTEXT,
                ),
            },
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "runtime-versions": {"python": "3.11"},
                        "commands": ["pip install boto3 --quiet"],
                    },
                    "pre_build": {
                        "commands": [
                            # Upload golden test set to S3 (idempotent — overwrites same object)
                            "aws s3 cp benchmarks/golden_test_set.jsonl "
                            "s3://$ARTIFACT_BUCKET/validation/golden_test_set.jsonl --quiet",
                        ]
                    },
                    "build": {
                        "commands": [
                            "python cdk/buildspec/validate_model.py",
                        ]
                    },
                },
                "artifacts": {"files": ["validation_result.json"]},
            }),
            timeout=cdk.Duration.hours(1),
        )

        # ── Outputs ────────────────────────────────────────────────────────
        CfnOutput(
            self, "ValidationProjectName",
            value=self.validation_project.project_name,
            export_name="ABGatewayValidationProjectName",
        )
        CfnOutput(
            self, "ArtifactBucketName",
            value=self.artifact_bucket.bucket_name,
            export_name="ABGatewayArtifactBucketName",
        )
        CfnOutput(
            self, "ValidationRoleArn",
            value=self.validation_role.role_arn,
            export_name="ABGatewayValidationRoleArn",
        )

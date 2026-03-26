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
        # Needs SageMaker (Batch Transform) + S3 (input/output) + Logs
        # Does NOT use SageMakerFullAccess — principle of least privilege.
        self.validation_role = iam.Role(
            self,
            "ValidationCodeBuildRole",
            role_name="ab-gateway-validation-codebuild-role",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
            inline_policies={
                "SageMakerBatchTransform": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            sid="SageMakerTransformPermissions",
                            actions=[
                                "sagemaker:CreateModel",
                                "sagemaker:CreateTransformJob",
                                "sagemaker:DescribeTransformJob",
                                "sagemaker:StopTransformJob",
                                "sagemaker:DeleteModel",
                                "sagemaker:DescribeModelPackage",
                            ],
                            resources=["*"],
                        ),
                        iam.PolicyStatement(
                            sid="PassRoleForSageMaker",
                            actions=["iam:PassRole"],
                            resources=[f"arn:aws:iam::{account}:role/*"],
                            conditions={
                                "StringEquals": {
                                    "iam:PassedToService": "sagemaker.amazonaws.com"
                                }
                            },
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
            },
            # Stub buildspec for Phase 4a — just echoes inputs and exits 0.
            # Phase 4c replaces this with the real validation script.
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "build": {
                        "commands": [
                            "echo 'Phase 4a stub — validation not yet implemented'",
                            "echo \"MODEL_PACKAGE_ARN=$MODEL_PACKAGE_ARN\"",
                            "echo \"VARIANT_NAME=$VARIANT_NAME\"",
                            "echo '{\"stub\": true, \"passed\": true}' > validation_result.json",
                        ]
                    }
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

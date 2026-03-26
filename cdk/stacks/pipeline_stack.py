import aws_cdk as cdk
from aws_cdk import Stack, pipelines
from constructs import Construct
from .model_deploy_stage import ModelDeployStage


class PipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx = self.node.try_get_context

        # Source: GitHub via CodeConnections (formerly CodeStar Connections)
        # The connection must be in AVAILABLE state before first cdk deploy.
        # Connection was activated manually in the AWS console.
        source = pipelines.CodePipelineSource.connection(
            f"{ctx('github_owner')}/{ctx('github_repo')}",
            ctx("github_branch"),
            connection_arn=ctx("connection_arn"),
        )

        # Self-mutating CDK pipeline.
        # On every git push to main:
        #   1. UpdatePipeline stage runs first — updates the pipeline itself
        #   2. Then deploys ModelDeployStage (validation + canary infrastructure)
        pipeline = pipelines.CodePipeline(
            self,
            "Pipeline",
            pipeline_name="ab-gateway-infra-pipeline",
            synth=pipelines.ShellStep(
                "Synth",
                input=source,
                install_commands=[
                    "npm install -g aws-cdk@2.170.0",
                    "cd cdk",
                    "pip install -r requirements.txt",
                ],
                commands=[
                    # Working directory is still repo root here.
                    # ShellStep runs install_commands then commands in sequence.
                    "cd cdk && cdk synth",
                ],
                primary_output_directory="cdk/cdk.out",
            ),
            docker_enabled_for_synth=False,
        )

        # Deploy the model deployment infrastructure as a CDK Stage.
        # This deploys ValidationStack and CanaryStack into the same account/region.
        pipeline.add_stage(
            ModelDeployStage(
                self,
                "ModelDeployInfra",
                env=cdk.Environment(
                    account=ctx("account"),
                    region=ctx("region"),
                ),
            )
        )

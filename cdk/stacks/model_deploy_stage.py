import aws_cdk as cdk
from constructs import Construct
from .validation_stack import ValidationStack
from .canary_stack import CanaryStack


class ModelDeployStage(cdk.Stage):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Stack 1: S3 bucket + CodeBuild project for golden dataset validation
        validation = ValidationStack(self, "ValidationStack")

        # Stack 2: Canary Lambda + EventBridge approval trigger
        canary = CanaryStack(self, "CanaryStack")

        # CanaryStack depends on ValidationStack because the Canary Lambda's
        # VALIDATION_PROJECT_NAME env var references the CodeBuild project
        # that ValidationStack creates. Deploy validation first.
        canary.add_dependency(validation)

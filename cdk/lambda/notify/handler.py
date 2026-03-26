"""
Publish an SNS notification.
Subject and message come from the event (set by Pass states upstream).
"""
import boto3
import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sns_client         = boto3.client("sns")
NOTIFICATION_TOPIC = os.environ["NOTIFICATION_TOPIC_ARN"]


def lambda_handler(event, context):
    subject = event.get("subject", "[AB Gateway] Canary deployment notification")
    message = event.get("message", json.dumps(event, indent=2, default=str))

    sns_client.publish(
        TopicArn=NOTIFICATION_TOPIC,
        Subject=subject[:100],   # SNS subject max 100 chars
        Message=message,
    )
    logger.info(f"SNS published: {subject}")
    return {**event, "notified": True}

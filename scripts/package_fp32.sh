#!/usr/bin/env bash
set -e

# Usage: ./scripts/package_fp32.sh <variant> <s3_bucket> <full_s3_input_uri>
# variant: bert | distilbert
# Example:
#   ./scripts/package_fp32.sh bert ab-gateway-artifacts \
#     s3://ab-gateway-artifacts/models/bert-fp32/huggingface-pytorch-training-.../output/model.tar.gz

VARIANT=${1:?Usage: $0 <bert|distilbert> <s3_bucket> <s3_input_uri>}
S3_BUCKET=${2:?Usage: $0 <bert|distilbert> <s3_bucket> <s3_input_uri>}
S3_INPUT=${3:?Usage: $0 <bert|distilbert> <s3_bucket> <s3_input_uri>}

if [ "$VARIANT" = "bert" ]; then
    S3_FINAL="s3://$S3_BUCKET/models/bert-fp32/model.tar.gz"
    WORK_DIR="./bert-fp32-unpacked"
elif [ "$VARIANT" = "distilbert" ]; then
    S3_FINAL="s3://$S3_BUCKET/models/distilbert-fp32/model.tar.gz"
    WORK_DIR="./distilbert-fp32-unpacked"
else
    echo "Unknown variant: $VARIANT. Use 'bert' or 'distilbert'."
    exit 1
fi

echo "==> Downloading raw training output from $S3_INPUT..."
aws s3 cp "$S3_INPUT" ./raw-model.tar.gz

echo "==> Unpacking..."
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
tar -xzf raw-model.tar.gz -C "$WORK_DIR/"

echo "==> Injecting inference script..."
mkdir -p "$WORK_DIR/code"
cp models/inference_fp32/inference.py     "$WORK_DIR/code/"
cp models/inference_fp32/requirements.txt "$WORK_DIR/code/"

echo "==> Repacking..."
FINAL_TAR="./${VARIANT}-fp32-final.tar.gz"
cd "$WORK_DIR" && tar -czvf "../$FINAL_TAR" . && cd ..

echo "==> Uploading final artifact to $S3_FINAL..."
aws s3 cp "$FINAL_TAR" "$S3_FINAL"

echo "==> Done. Final artifact at $S3_FINAL"
rm -f raw-model.tar.gz "$FINAL_TAR"

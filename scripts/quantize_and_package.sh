#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./scripts/quantize_and_package.sh <model_dir_or_s3_path> <s3_bucket>

INPUT_PATH=${1:?Usage: $0 <model_dir_or_s3_path> <s3_bucket>}
S3_BUCKET=${2:?Usage: $0 <model_dir_or_s3_path> <s3_bucket>}

OUTPUT_DIR="./bert-int8-onnx"
ARTIFACT_DIR="./bert-int8-artifact"
TMP_DIR="./tmp-model"
EXTRACT_DIR="./tmp-model-extracted"
LOG_FILE="./pipeline.log"

# --- Logging function ---
log() {
  local msg="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$LOG_FILE"
}

# --- Start ---
log "Starting pipeline"
log "Input path: $INPUT_PATH"
log "S3 bucket: $S3_BUCKET"

rm -rf "$TMP_DIR" "$EXTRACT_DIR" "$OUTPUT_DIR" "$ARTIFACT_DIR" "$LOG_FILE"
mkdir -p "$TMP_DIR" "$EXTRACT_DIR"

# --- Handle S3 input ---
if [[ "$INPUT_PATH" == s3://* ]]; then
  log "Detected S3 input"

  log "Downloading from S3..."
  aws s3 cp "$INPUT_PATH" "$TMP_DIR/input.tar.gz"

  log "Checking if input is a tar.gz..."
  if file "$TMP_DIR/input.tar.gz" | grep -q "gzip compressed"; then
    log "Extracting model archive..."
    tar -xzf "$TMP_DIR/input.tar.gz" -C "$EXTRACT_DIR"
    MODEL_DIR="$EXTRACT_DIR"
  else
    log "Input is not an archive, using as directory"
    MODEL_DIR="$TMP_DIR"
  fi
else
  log "Using local model directory"
  MODEL_DIR="$INPUT_PATH"
fi

log "Model directory resolved: $MODEL_DIR"

# --- Validate model ---
if [ ! -f "$MODEL_DIR/config.json" ]; then
  log "ERROR: config.json not found"
  log "Dumping directory contents:"
  ls -R "$MODEL_DIR" | tee -a "$LOG_FILE"
  exit 1
fi

log "Model validation passed"

# --- Quantization ---
log "Starting quantization..."
python models/quantize.py \
  --model_dir "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

log "Quantization complete"
log "Output files:"
ls -R "$OUTPUT_DIR" | tee -a "$LOG_FILE"

# --- Packaging ---
log "Packaging artifact..."
mkdir -p "$ARTIFACT_DIR/code"

cp "$OUTPUT_DIR"/model.onnx "$ARTIFACT_DIR/" 2>/dev/null || log "WARNING: model.onnx missing"
cp "$OUTPUT_DIR"/config.json "$ARTIFACT_DIR/" 2>/dev/null || true
cp "$OUTPUT_DIR"/tokenizer_config.json "$ARTIFACT_DIR/" 2>/dev/null || true
cp "$OUTPUT_DIR"/tokenizer.json "$ARTIFACT_DIR/" 2>/dev/null || true
cp "$OUTPUT_DIR"/vocab.txt "$ARTIFACT_DIR/" 2>/dev/null || true
cp "$OUTPUT_DIR"/special_tokens_map.json "$ARTIFACT_DIR/" 2>/dev/null || true

cp models/inference_onnx/inference.py "$ARTIFACT_DIR/code/"
cp models/inference_onnx/requirements.txt "$ARTIFACT_DIR/code/"

cd "$ARTIFACT_DIR"
tar -czvf ../bert-int8.tar.gz . | tee -a "../$LOG_FILE"
cd ..

log "Artifact created: bert-int8.tar.gz"

# --- Upload ---
log "Uploading to S3..."
aws s3 cp bert-int8.tar.gz "s3://$S3_BUCKET/models/bert-int8/model.tar.gz" \
  2>&1 | tee -a "$LOG_FILE"

# --- Cleanup ---
log "Cleaning up temp directories..."
rm -rf "$TMP_DIR" "$EXTRACT_DIR"

log "Pipeline completed successfully"
"""
03_op_check.py
Definitively confirm whether the ONNX model contains INT8 operations.
Run from repo root: python audit/03_op_check.py
"""
import sys
import os
from collections import Counter

print("=== 03_op_check.py ===")

MODEL_PATH = "./bert-int8-onnx/model.onnx"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Expected INT8 ONNX model at ./bert-int8-onnx/model.onnx")
    sys.exit(1)

try:
    import onnx
except ImportError:
    print("ERROR: onnx is not installed. Run: pip install onnx")
    sys.exit(1)

# --- Size check ---
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"\nModel size on disk: {model_size_mb:.1f} MB")
print(f"  Expected ~110 MB for valid INT8 BERT")
print(f"  Expected ~420 MB for FP32 BERT ONNX")

if model_size_mb > 300:
    print(f"  SIZE FLAG: {model_size_mb:.0f} MB suggests this may be an FP32 model.")
elif model_size_mb < 80:
    print(f"  SIZE FLAG: {model_size_mb:.0f} MB is unusually small — check model integrity.")
else:
    print(f"  SIZE OK: {model_size_mb:.0f} MB is consistent with INT8 quantization.")

fp32_backup = "./bert-int8-onnx/model_fp32.onnx"
if os.path.exists(fp32_backup):
    fp32_mb = os.path.getsize(fp32_backup) / (1024 * 1024)
    ratio = fp32_mb / model_size_mb
    print(f"\nFP32 backup (model_fp32.onnx): {fp32_mb:.1f} MB")
    print(f"Compression ratio: {ratio:.2f}x  (expected ~4x for INT8)")

# --- Graph op analysis ---
print("\nLoading ONNX graph (this may take a moment for large models)...")
model = onnx.load(MODEL_PATH)

QUANTIZED_OPS = {
    "MatMulInteger",
    "QLinearMatMul",
    "QGemm",
    "QuantizeLinear",
    "DequantizeLinear",
    "QLinearConv",
}

op_counts = {}
for node in model.graph.node:
    op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

found_quantized = [op for op in op_counts if op in QUANTIZED_OPS]
total_nodes = sum(op_counts.values())

print("\n=== ONNX Graph Op Analysis ===")
print(f"Total nodes: {total_nodes}")
print(f"\nTop 15 op types:")
for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
    marker = " <-- INT8 quantized" if op in QUANTIZED_OPS else ""
    print(f"  {op:<35} {count:>5}{marker}")

print(f"\nQuantized ops present: {len(found_quantized) > 0}")
if found_quantized:
    quant_counts = {op: op_counts[op] for op in found_quantized}
    for op, count in sorted(quant_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")
    print("\n  VERDICT: Model is correctly quantized — INT8 ops are present in the graph.")
    print("  The 4.6% speedup gap at the endpoint is NOT caused by missing quantization.")
    print("  Likely explanation: SageMaker routing/serialization overhead or limited")
    print("  AVX-512 VNNI support on ml.m5.large (older Skylake/Cascade Lake generation).")
else:
    print("  NONE — this model is running as FP32")
    print("\n  VERDICT: RED FLAG — model has no INT8 ops despite being named 'int8'.")
    print("  Likely cause: quantize.py ran but model_quantized.onnx was not renamed,")
    print("  or the FP32 model.onnx was packaged instead of the quantized output.")
    print("  Fix: re-run models/quantize.py and verify:")
    print("    1. model_quantized.onnx is produced (~110 MB)")
    print("    2. It is renamed to model.onnx before packaging into bert-int8.tar.gz")

# --- Check graph initializers for INT8 weight tensors ---
print("\n=== Weight Tensor Data Types ===")
dtype_map = {
    1: "FLOAT32",
    2: "UINT8",
    3: "INT8",
    6: "INT32",
    7: "INT64",
    10: "FLOAT16",
    11: "FLOAT64",
}
dtype_counts = {}
for init in model.graph.initializer:
    dtype_name = dtype_map.get(init.data_type, f"type_{init.data_type}")
    dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
    print(f"  {dtype:<12} {count:>5} initializers")

int8_count = dtype_counts.get("INT8", 0) + dtype_counts.get("UINT8", 0)
if int8_count > 0:
    print(f"\n  INT8/UINT8 weight tensors found ({int8_count} total) — confirms quantization.")
else:
    print(f"\n  No INT8/UINT8 weight tensors found — model weights are all FP32.")

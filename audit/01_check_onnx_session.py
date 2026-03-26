"""
01_check_onnx_session.py
Inspect the ORT session config and graph for Variant B (BERT INT8).
Run from repo root: python audit/01_check_onnx_session.py
"""
import sys
import os
from collections import Counter

print("=== 01_check_onnx_session.py ===")

MODEL_PATH = "./bert-int8-onnx/model.onnx"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Expected INT8 ONNX model at ./bert-int8-onnx/model.onnx")
    sys.exit(1)

model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"Model file: {MODEL_PATH}  ({model_size_mb:.1f} MB)")

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is not installed. Run: pip install onnxruntime")
    sys.exit(1)

try:
    import onnx
except ImportError:
    print("ERROR: onnx is not installed. Run: pip install onnx")
    sys.exit(1)

# --- Session inspection ---
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    MODEL_PATH,
    sess_options,
    providers=["CPUExecutionProvider"],
)

print("\n=== ORT Session Config ===")
print(f"Providers:              {session.get_providers()}")
print(f"Graph optimization:     {sess_options.graph_optimization_level}")
print(f"Intra-op threads:       {sess_options.intra_op_num_threads}")
print(f"Inter-op threads:       {sess_options.inter_op_num_threads}")
print(f"Model size on disk:     {model_size_mb:.1f} MB")
print(f"  (expected ~110 MB for INT8, ~420 MB for FP32)")

fp32_backup = "./bert-int8-onnx/model_fp32.onnx"
if os.path.exists(fp32_backup):
    fp32_size_mb = os.path.getsize(fp32_backup) / (1024 * 1024)
    compression = fp32_size_mb / model_size_mb
    print(f"FP32 backup size:       {fp32_size_mb:.1f} MB")
    print(f"Compression ratio:      {compression:.2f}x (expected ~4x for INT8)")

# --- Graph inspection ---
print("\n=== ONNX Graph Inspection ===")
model = onnx.load(MODEL_PATH)

QUANTIZED_OPS = {
    "QLinearMatMul", "MatMulInteger", "QGemm", "QLinearConv",
    "QuantizeLinear", "DequantizeLinear",
}

op_counts = Counter(node.op_type for node in model.graph.node)
found_quantized = [op for op in op_counts if op in QUANTIZED_OPS]

total_nodes = sum(op_counts.values())
print(f"Total nodes: {total_nodes}")
print("\nTop 10 op types by frequency:")
for op, count in op_counts.most_common(10):
    marker = " <-- QUANTIZED" if op in QUANTIZED_OPS else ""
    print(f"  {op:<35} {count:>5}{marker}")

print(f"\nQuantized ops found: {bool(found_quantized)}")
if found_quantized:
    quant_detail = {op: op_counts[op] for op in found_quantized}
    for op, count in sorted(quant_detail.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")
    print("\n  STATUS: Model appears correctly quantized (INT8 ops present).")
else:
    print("  NONE — this model is running as FP32")
    print("  Fix: re-run models/quantize.py and verify model_quantized.onnx is")
    print("       renamed to model.onnx before packaging.")
    print("\n  STATUS: RED FLAG — model has no quantized ops despite .onnx being loaded.")

# --- Input/output info ---
print("\n=== Model Inputs/Outputs ===")
for inp in session.get_inputs():
    print(f"  Input:  {inp.name}  shape={inp.shape}  type={inp.type}")
for out in session.get_outputs():
    print(f"  Output: {out.name}  shape={out.shape}  type={out.type}")

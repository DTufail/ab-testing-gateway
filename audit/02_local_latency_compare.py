"""
02_local_latency_compare.py
Local latency: FP32 PyTorch vs INT8 ONNX (no network hop).
Run from repo root: python audit/02_local_latency_compare.py
"""
import sys
import os
import time
import statistics

print("=== 02_local_latency_compare.py ===")

WARM_UP = 100
TIMED   = 200
SEQ_LEN = 128

# Model paths — checked in order; first existing path wins
BERT_FP32_CANDIDATES = [
    "./bert-fp32-finetuned",
    "./output",          # training output dir used in this repo
]
BERT_INT8_PATH  = "./bert-int8-onnx/model.onnx"
DISTILBERT_CANDIDATES = [
    "./distilbert-fp32-finetuned",
    "./distilbert-output",
]

TEST_QUERY = "What is my card limit?"


def find_dir(candidates, label):
    for p in candidates:
        if os.path.isdir(p):
            return p
    print(f"ERROR: {label} directory not found.")
    print(f"  Searched: {candidates}")
    print(f"  Create a symlink or copy the model to one of the above paths.")
    return None


def percentile(data, p):
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_pytorch(model_dir, label, warm_up, timed):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        print(f"ERROR: Missing dependency for {label}: {e}")
        return None

    print(f"\nLoading {label} from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(
        TEST_QUERY,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN,
    )

    print(f"  Warming up ({warm_up} requests)...")
    with torch.no_grad():
        for _ in range(warm_up):
            model(**inputs)

    print(f"  Timing ({timed} requests)...")
    times = []
    with torch.no_grad():
        for _ in range(timed):
            t0 = time.perf_counter()
            model(**inputs)
            times.append((time.perf_counter() - t0) * 1000)

    return times


def run_onnx(model_path, label, warm_up, timed):
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"ERROR: Missing dependency for {label}: {e}")
        return None

    print(f"\nLoading {label} from {model_path} ...")
    # Session with full graph optimisation
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=["CPUExecutionProvider"],
    )

    tokenizer_dir = os.path.dirname(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    inputs_raw = tokenizer(
        TEST_QUERY,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN,
    )
    input_feed = {k: v for k, v in inputs_raw.items()}

    print(f"  Warming up ({warm_up} requests)...")
    for _ in range(warm_up):
        session.run(["logits"], input_feed)

    print(f"  Timing ({timed} requests)...")
    times = []
    for _ in range(timed):
        t0 = time.perf_counter()
        session.run(["logits"], input_feed)
        times.append((time.perf_counter() - t0) * 1000)

    return times


def fmt(ms):
    return f"{ms:.1f}ms"


# --- Load and time each model ---
results = {}
errors  = []

# Variant A — BERT FP32 (PyTorch)
bert_fp32_dir = find_dir(BERT_FP32_CANDIDATES, "BERT FP32")
if bert_fp32_dir:
    times = run_pytorch(bert_fp32_dir, "BERT FP32", WARM_UP, TIMED)
    if times:
        results["BERT FP32"] = times
    else:
        errors.append("BERT FP32")
else:
    errors.append("BERT FP32 (dir not found)")

# Variant B — BERT INT8 (ONNX)
if not os.path.exists(BERT_INT8_PATH):
    print(f"ERROR: BERT INT8 model not found at {BERT_INT8_PATH}")
    errors.append("BERT INT8 (model.onnx not found)")
else:
    times = run_onnx(BERT_INT8_PATH, "BERT INT8", WARM_UP, TIMED)
    if times:
        results["BERT INT8"] = times
    else:
        errors.append("BERT INT8")

# Variant C — DistilBERT FP32 (PyTorch)
distilbert_dir = find_dir(DISTILBERT_CANDIDATES, "DistilBERT FP32")
if distilbert_dir:
    times = run_pytorch(distilbert_dir, "DistilBERT", WARM_UP, TIMED)
    if times:
        results["DistilBERT"] = times
    else:
        errors.append("DistilBERT")
else:
    errors.append("DistilBERT (dir not found)")

if not results:
    print("\nERROR: No models could be loaded. Cannot produce latency comparison.")
    sys.exit(1)

# --- Report ---
print(f"\n=== Local Latency ({WARM_UP} warm-up + {TIMED} timed, batch=1, seq_len={SEQ_LEN}) ===")

header = f"{'Variant':<16} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8} {'speedup_vs_A':>14}"
print(header)
print("-" * len(header))

fp32_p50 = None
rows = {}
for label in ["BERT FP32", "BERT INT8", "DistilBERT"]:
    if label not in results:
        print(f"{label:<16}  (not available — {label} missing)")
        continue
    t = results[label]
    n = len(t)
    p50  = statistics.median(t)
    p95  = sorted(t)[int(0.95 * n)]
    p99  = sorted(t)[int(0.99 * n)]
    mean = statistics.mean(t)
    rows[label] = {"p50": p50, "p95": p95, "p99": p99, "mean": mean}
    if label == "BERT FP32":
        fp32_p50 = p50
    speedup = f"{fp32_p50 / p50:.2f}x" if fp32_p50 and p50 > 0 else "N/A"
    print(f"{label:<16} {fmt(p50):>8} {fmt(p95):>8} {fmt(p99):>8} {fmt(mean):>8} {speedup:>14}")

if errors:
    print(f"\nSkipped (not available): {', '.join(errors)}")

# --- Diagnosis ---
print("\nDiagnosis:")
if "BERT FP32" in rows and "BERT INT8" in rows:
    speedup = rows["BERT FP32"]["p50"] / rows["BERT INT8"]["p50"]
    print(f"  INT8 speedup vs FP32: {speedup:.2f}x  (healthy range: 1.5x–3.5x on AVX2/AVX-512)")
    if speedup < 1.2:
        print("  → speedup < 1.2x: quantization is NOT effective on this CPU.")
        print("    Possible causes: no AVX-512 VNNI support, or model not truly quantized.")
    elif speedup < 1.5:
        print("  → speedup 1.2x–1.5x: modest gain, may indicate limited VNNI support")
        print("    on this CPU generation. Check with lscpu for AVX-512 VNNI flag.")
    else:
        print("  → speedup > 1.5x: INT8 is working. Gap in endpoint numbers likely")
        print("    explained by SageMaker serialisation + routing overhead (~XYZms).")
else:
    print("  Cannot compute INT8 speedup — one or both models unavailable locally.")
    print("  Missing models:", [m for m in ["BERT FP32", "BERT INT8"] if m not in rows])

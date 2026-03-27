"""
diagnose_accuracy.py — Diagnose the 73% accuracy on VariantB-BERT-INT8.

Runs 5 golden-set examples against both variants side-by-side and prints
enough detail to distinguish three failure modes:
  1. Real INT8 quantization degradation  → FP32 ~90%, INT8 ~73%
  2. Label mismatch in golden set        → both variants score ~73%
  3. String vs int type mismatch         → predicted_id type != label type

Run from repo root:
    python scripts/diagnose_accuracy.py

Reads:
    benchmarks/endpoint_name.txt   — live endpoint name
    models/golden_test_set.jsonl   — golden test set (local copy)
"""
import json
import pathlib
import sys
from collections import Counter
import boto3

# ── Paths (relative to repo root, i.e. the directory containing this script's parent) ──
REPO_ROOT       = pathlib.Path(__file__).parent.parent
ENDPOINT_FILE   = REPO_ROOT / "benchmarks" / "endpoint_name.txt"
GOLDEN_SET_FILE = REPO_ROOT / "models" / "golden_test_set.jsonl"

VARIANT_A = "VariantA-BERT-FP32"
VARIANT_B = "VariantB-BERT-INT8"
N_EXAMPLES = 5

runtime = boto3.client("sagemaker-runtime")


def load_endpoint_name() -> str:
    name = ENDPOINT_FILE.read_text().strip()
    if not name:
        sys.exit(f"ERROR: {ENDPOINT_FILE} is empty")
    return name


def load_examples(n: int) -> list[dict]:
    lines = GOLDEN_SET_FILE.read_text().splitlines()
    examples = [json.loads(l) for l in lines if l.strip()]
    return examples[:n]


def invoke(endpoint_name: str, variant: str, example: dict) -> dict:
    body = json.dumps({"inputs": example["text"]})
    try:
        resp = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            TargetVariant=variant,
            ContentType="application/json",
            Accept="application/json",
            Body=body,
        )
        result = json.loads(resp["Body"].read().decode("utf-8"))
        if isinstance(result, list):
            result = result[0]

        predicted_id    = result.get("predicted_id")
        predicted_label = result.get("predicted_label", "N/A")
        confidence      = result.get("confidence", "N/A")
        label           = example["label"]

        # Type check — the critical comparison in validate_model.py is ==
        type_ok  = type(predicted_id) is type(label)
        correct  = predicted_id == label

        return {
            "correct":          correct,
            "predicted_id":     predicted_id,
            "predicted_id_type": type(predicted_id).__name__,
            "predicted_label":  predicted_label,
            "confidence":       confidence,
            "label":            label,
            "label_type":       type(label).__name__,
            "label_name":       example.get("label_name", "N/A"),
            "type_match":       type_ok,
            "error":            None,
        }
    except Exception as exc:
        return {
            "correct":          False,
            "predicted_id":     None,
            "predicted_id_type": "N/A",
            "predicted_label":  "N/A",
            "confidence":       "N/A",
            "label":            example["label"],
            "label_type":       type(example["label"]).__name__,
            "label_name":       example.get("label_name", "N/A"),
            "type_match":       False,
            "error":            str(exc),
        }


def print_variant_results(variant: str, results: list[dict]) -> int:
    correct_count = sum(1 for r in results if r["correct"])
    print(f"\n{'='*70}")
    print(f"  {variant}   ({correct_count}/{len(results)} correct)")
    print(f"{'='*70}")
    print(f"  {'#':<3}  {'correct':<8}  {'pred_id':<9}  {'pred_type':<10}  "
          f"{'label':<7}  {'lbl_type':<9}  {'type_match':<11}  {'confidence':<11}  pred_label")
    print(f"  {'-'*3}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*7}  {'-'*9}  {'-'*11}  {'-'*11}  {'-'*20}")
    for i, r in enumerate(results, 1):
        if r["error"]:
            print(f"  {i:<3}  ERROR: {r['error']}")
        else:
            print(
                f"  {i:<3}  {str(r['correct']):<8}  {str(r['predicted_id']):<9}  "
                f"{r['predicted_id_type']:<10}  {str(r['label']):<7}  "
                f"{r['label_type']:<9}  {str(r['type_match']):<11}  "
                f"{str(r['confidence']):<11}  {r['predicted_label']}"
            )
    return correct_count


def diagnose(fp32_results: list[dict], int8_results: list[dict]) -> None:
    fp32_correct = sum(1 for r in fp32_results if r["correct"])
    int8_correct = sum(1 for r in int8_results if r["correct"])
    n = len(fp32_results)

    any_type_mismatch_fp32 = any(not r["type_match"] for r in fp32_results if not r["error"])
    any_type_mismatch_int8 = any(not r["type_match"] for r in int8_results if not r["error"])

    print(f"\n{'='*70}")
    print("  DIAGNOSIS")
    print(f"{'='*70}")
    print(f"  FP32 correct: {fp32_correct}/{n}")
    print(f"  INT8 correct: {int8_correct}/{n}")

    # Type mismatch check
    if any_type_mismatch_fp32 or any_type_mismatch_int8:
        print("\n  [!] TYPE MISMATCH DETECTED")
        print("      predicted_id and label have different Python types.")
        print("      Even when the numeric value is the same, == returns False.")
        print("      Fix: cast predicted_id to int in validate_model.py:")
        print("           predicted_id = int(result.get('predicted_id'))")
    else:
        print("\n  [OK] No type mismatch — both predicted_id and label are the same type.")

    # Accuracy pattern check
    print()
    if fp32_correct == int8_correct:
        print("  [!] BOTH variants score the same.")
        print("      INT8 quantization is NOT the cause of low accuracy.")
        print("      Likely cause: label IDs in golden_test_set.jsonl don't match")
        print("      the model's id2label mapping. Check config.json id2label vs")
        print("      the label integers in models/golden_test_set.jsonl.")
    elif fp32_correct > int8_correct:
        print("  [!] FP32 outperforms INT8.")
        print("      INT8 quantization is the likely cause of accuracy degradation.")
        print("      Next step: run full 200-example validation on VariantA-BERT-FP32")
        print("      to confirm the FP32 baseline is ~90%, then decide whether")
        print("      INT8 accuracy drop is acceptable.")
    else:
        print("  [?] INT8 outperforms FP32 on this sample — inconclusive.")
        print("      Run full 200-example validation on both variants to get")
        print("      statistically meaningful numbers.")

    # Confusion pairs
    for label, results in (("FP32", fp32_results), ("INT8", int8_results)):
        errors = [(r["label"], r["predicted_id"]) for r in results if not r["correct"] and not r["error"]]
        if errors:
            top = Counter(errors).most_common(10)
            print(f"  {label} confusion pairs (true_label → predicted_id):")
            for (true_id, pred_id), count in top:
                print(f"    label {true_id} → predicted {pred_id}  (x{count})")
        else:
            print(f"  {label}: no errors in this sample.")

    print()
    print("  NOTE: This sample (5 examples) is indicative only.")
    print("  Run the full validation with validate_model.py for conclusive results.")
    print()


def main() -> None:
    endpoint_name = load_endpoint_name()
    print(f"Endpoint : {endpoint_name}")
    print(f"Variants : {VARIANT_A}  vs  {VARIANT_B}")

    examples = load_examples(N_EXAMPLES)
    print(f"Examples : {len(examples)} loaded from {GOLDEN_SET_FILE.relative_to(REPO_ROOT)}")

    print("\nInvoking VariantA-BERT-FP32 ...")
    fp32_results = [invoke(endpoint_name, VARIANT_A, ex) for ex in examples]

    print("Invoking VariantB-BERT-INT8 ...")
    int8_results = [invoke(endpoint_name, VARIANT_B, ex) for ex in examples]

    print_variant_results(VARIANT_A, fp32_results)
    print_variant_results(VARIANT_B, int8_results)
    diagnose(fp32_results, int8_results)


if __name__ == "__main__":
    main()

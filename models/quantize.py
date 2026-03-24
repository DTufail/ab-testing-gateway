"""
Quantize fine-tuned BERT FP32 to ONNX INT8 using HuggingFace Optimum.
Produces Variant B (VariantB-BERT-INT8).
"""
import argparse
import os
import shutil
import numpy as np
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Export BERT to ONNX and quantize to INT8")
    parser.add_argument("--model_dir", default="./bert-fp32-finetuned",
                        help="Path to fine-tuned BERT FP32 model directory")
    parser.add_argument("--output_dir", default="./bert-int8-onnx",
                        help="Directory to write quantized ONNX model")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Export FP32 model to ONNX via Optimum
    print(f"Exporting {model_dir} to ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    ort_model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(output_dir)

    fp32_onnx_path = Path(output_dir) / "model.onnx"
    fp32_size_mb = fp32_onnx_path.stat().st_size / (1024 * 1024)
    print(f"FP32 ONNX size: {fp32_size_mb:.1f} MB")

    # Step 2: Quantize to INT8 using Optimum
    print("Quantizing to INT8 (dynamic, avx512_vnni config)...")
    quantizer = ORTQuantizer.from_pretrained(output_dir)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=output_dir, quantization_config=dqconfig)

    # Step 3: Rename model_quantized.onnx → model.onnx
    quantized_path = Path(output_dir) / "model_quantized.onnx"
    final_path = Path(output_dir) / "model.onnx"

    if quantized_path.exists():
        # Back up original FP32 ONNX first
        fp32_backup = Path(output_dir) / "model_fp32.onnx"
        if final_path.exists():
            shutil.move(str(final_path), str(fp32_backup))
            fp32_size_mb = fp32_backup.stat().st_size / (1024 * 1024)
            print(f"FP32 ONNX size: {fp32_size_mb:.1f} MB (saved as model_fp32.onnx)")
        shutil.move(str(quantized_path), str(final_path))
        print(f"Renamed model_quantized.onnx → model.onnx")
    else:
        raise FileNotFoundError(
            f"Expected quantized model at {quantized_path} but it was not found. "
            "Check that quantization completed successfully."
        )

    int8_size_mb = final_path.stat().st_size / (1024 * 1024)
    print(f"INT8 ONNX size: {int8_size_mb:.1f} MB")
    if fp32_size_mb > 0:
        print(f"Compression ratio: {fp32_size_mb / int8_size_mb:.2f}x")

    # Step 4: Validate — run one dummy inference and assert output shape
    print("\nValidating ONNX model with dummy inference...")
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(final_path),
        providers=["CPUExecutionProvider"],
    )

    tokenizer_check = AutoTokenizer.from_pretrained(output_dir)
    dummy_inputs = tokenizer_check(
        "I lost my card",
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    input_feed = {k: v for k, v in dummy_inputs.items()}
    outputs = session.run(["logits"], input_feed)
    logits = outputs[0]

    assert logits.shape == (1, 77), (
        f"Expected output shape (1, 77) but got {logits.shape}. "
        "Check that the model was trained on BANKING77 (77 classes)."
    )

    predicted_id = int(np.argmax(logits, axis=-1)[0])
    print(f"Dummy inference passed. Predicted class ID: {predicted_id}")
    print("\nQuantization complete.")
    print(f"Output directory: {output_dir}/")
    print(f"  model.onnx          — INT8 quantized model")
    print(f"  model_fp32.onnx     — original FP32 ONNX (for size comparison)")


if __name__ == "__main__":
    main()

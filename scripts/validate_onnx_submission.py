"""CLI tool for students and organizers to validate ONNX submission models (1080p AoT).

Validates tensor names, shapes ([1, 3, 1080, 1920] -> [1, 1080, 1920]), dtypes, model size, and opset.
Executes the latency benchmark protocol (20 warm-up passes, 30 timed passes, median p50).
Optionally evaluates predictions and Foreground Micro-mIoU against a validation ground truth split.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.gt_parser import (
    NUM_ACTIVE_CLASSES,
    TAXONOMY_NAMES,
    load_ground_truth_masks,
)
from src.evaluation.onnx_runner import (
    ALLOWED_OPSETS,
    EXPECTED_INPUT_NAME,
    EXPECTED_INPUT_SHAPE,
    EXPECTED_OUTPUT_NAME,
    EXPECTED_OUTPUT_SHAPE,
    MAX_MODEL_SIZE_BYTES,
    OnnxModelRunner,
    validate_onnx_contract,
)
from src.metrics.segmentation_metrics import (
    DatasetEvaluationResult,
    aggregate_dataset_metrics,
    compute_pixel_distribution,
    compute_segmentation_metrics,
)


def _rel_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return p.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official AoT 1080p ONNX submission contract and latency validator."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to student .onnx submission model.",
    )
    parser.add_argument(
        "--val-gt",
        type=Path,
        default=None,
        help="Optional path to validation ground truth ZIP or masks directory.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=ROOT / "data" / "val_in_domain" / "images",
        help="Directory containing source RGB 1080p keyframes matching ground truth stems.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Validation split name (default: all).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warm-up passes for latency benchmarking (default: 20).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of timed passes for latency benchmarking (default: 30).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save JSON validation report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()

    print("=" * 70)
    print("OFFICIAL AOT 1080P ONNX SUBMISSION VALIDATOR")
    print("=" * 70)
    print(f"Model: {_rel_path(model_path)}")

    # 1. Contract validation
    report = validate_onnx_contract(model_path, raise_on_error=False)

    print("\n--- CONTRACT VERIFICATION ---")
    status_str = "PASSED" if report.is_valid else "FAILED"
    print(f"Status:        {status_str}")
    print(
        f"File Size:     {report.file_size_mb:.2f} MB (limit: {MAX_MODEL_SIZE_BYTES / (1024 * 1024):.1f} MB)"
    )
    print(f"Opset:         {report.opset_version} (allowed: {ALLOWED_OPSETS})")
    print(f"Input:         '{report.input_name}' {report.input_shape} {report.input_type}")
    print(f"Expected In:   '{EXPECTED_INPUT_NAME}' {EXPECTED_INPUT_SHAPE} tensor(uint8)")
    print(f"Output:        '{report.output_name}' {report.output_shape} {report.output_type}")
    print(f"Expected Out:  '{EXPECTED_OUTPUT_NAME}' {EXPECTED_OUTPUT_SHAPE} tensor(uint8)")

    if not report.is_valid:
        print("\nCONTRACT ERRORS:")
        for err in report.errors:
            print(f"  [X] {err}")
        return 1

    # 2. Latency benchmarking
    print(f"\n--- LATENCY BENCHMARK ({args.warmup} warm-up, {args.runs} timed runs) ---")
    try:
        runner = OnnxModelRunner(model_path, validate_contract=False)
        bench = runner.benchmark_latency(warmup_runs=args.warmup, timed_runs=args.runs)
        print(f"Median (p50):  {bench.p50_ms:.2f} ms")
        print(f"p90:           {bench.p90_ms:.2f} ms")
        print(f"Mean:          {bench.mean_ms:.2f} ms (+/- {bench.std_ms:.2f} ms)")
        print(f"Min / Max:     {bench.min_ms:.2f} ms / {bench.max_ms:.2f} ms")
    except Exception as exc:
        print(f"Error executing latency benchmark: {exc}")
        return 1

    # 3. Validation ground truth evaluation (optional)
    eval_result: DatasetEvaluationResult | None = None
    if args.val_gt is not None and args.val_gt.exists():
        gt_path = args.val_gt.resolve()
        print(f"\n--- EVALUATING AGAINST {_rel_path(gt_path)} (Split: {args.split}) ---")
        try:
            gt_masks = load_ground_truth_masks(gt_path, split=args.split)
            frame_metrics = {}
            pred_dists = {}
            gt_dists = {}

            images_dir = args.images_dir.resolve()
            found_frames = 0

            for stem, gt_mask in gt_masks.items():
                img_path = None
                for ext in (".png", ".jpg", ".jpeg"):
                    candidate = images_dir / f"{stem}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break

                if img_path is None:
                    print(
                        f"Warning: Image file for stem '{stem}' not found in {_rel_path(images_dir)}. Skipping."
                    )
                    continue

                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"Warning: Failed to read image '{_rel_path(img_path)}'. Skipping.")
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                pred_mask = runner.predict(rgb)
                found_frames += 1

                m = compute_segmentation_metrics(
                    pred=pred_mask,
                    target=gt_mask,
                    num_classes=NUM_ACTIVE_CLASSES,
                    ignore_index=255,
                    class_names=TAXONOMY_NAMES,
                )
                frame_metrics[stem] = m
                pred_dists[stem] = compute_pixel_distribution(pred_mask, NUM_ACTIVE_CLASSES)
                gt_dists[stem] = compute_pixel_distribution(gt_mask, NUM_ACTIVE_CLASSES)

            if found_frames == 0:
                print("Error: No matching keyframe images could be found for evaluation.")
            else:
                eval_result = aggregate_dataset_metrics(
                    frame_metrics=frame_metrics,
                    num_classes=NUM_ACTIVE_CLASSES,
                    class_names=TAXONOMY_NAMES,
                    pixel_distribution_pred=pred_dists,
                    pixel_distribution_gt=gt_dists,
                )

                print("\nVALIDATION METRICS:")
                print(f"Frames Evaluated:            {found_frames}/{len(gt_masks)}")
                print(f"Foreground Micro-mIoU (1,2): {eval_result.mean_iou:.4f}")
                print(f"All-Classes Micro-mIoU:      {eval_result.all_classes_mean_iou:.4f}")
                print(f"Titan Count MAE:             {eval_result.titan_count_mae:.4f}")
                print(f"Pixel Accuracy:              {eval_result.pixel_accuracy * 100:.2f} %")
                print("\nClass-wise IoU:")
                for c_name, iou_val in eval_result.iou_per_class.items():
                    val_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "N/A (No GT support)"
                    print(f"  {c_name:<12}: {val_str}")

        except Exception as exc:
            print(f"Error during validation evaluation: {exc}")
            return 1

    # 4. Optional JSON export
    if args.output_json is not None:
        out_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": _rel_path(model_path),
            "contract": {
                "is_valid": report.is_valid,
                "file_size_bytes": report.file_size_bytes,
                "opset_version": report.opset_version,
                "input_name": report.input_name,
                "input_shape": list(report.input_shape),
                "output_name": report.output_name,
                "output_shape": list(report.output_shape),
                "errors": report.errors,
            },
            "latency": {
                "p50_ms": bench.p50_ms,
                "p90_ms": bench.p90_ms,
                "mean_ms": bench.mean_ms,
                "std_ms": bench.std_ms,
                "min_ms": bench.min_ms,
                "max_ms": bench.max_ms,
                "warmup_runs": bench.warmup_runs,
                "timed_runs": bench.timed_runs,
            },
            "metrics": eval_result.to_dict() if eval_result is not None else None,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nReport saved to: {_rel_path(args.output_json)}")

    print("\nResult: SUBMISSION COMPLIES WITH OFFICIAL CONTRACT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""Semantic segmentation evaluation metrics for Attack on Titan (1080p).

Primary Competition Metric:
- Foreground Micro-mIoU across active foreground classes {1: Pure_Titan, 2: Human}.
  False positives predicted on Background (0) directly increase FP_1 / FP_2 and reduce IoU.
  A trivial model predicting all zeros (Background) scores 0.0000 on Foreground Micro-mIoU.

Diagnostic Metrics:
- All-Classes Micro-mIoU across {0: Background, 1: Pure_Titan, 2: Human}.
- Connected-component Pure_Titan count and Count MAE.
"""

from dataclasses import dataclass
import cv2
import numpy as np


CLASS_NAMES_DEFAULT: dict[int, str] = {
    0: "Background",
    1: "Pure_Titan",
    2: "Human",
}

FOREGROUND_CLASS_IDS: tuple[int, ...] = (1, 2)
DEFAULT_BOUNDARY_TOLERANCE_PX: int = 3


def count_class_instances(
    mask: np.ndarray,
    class_id: int = 1,
    min_area_px: int = 64,
) -> int:
    """Count connected components of a target class with area >= min_area_px."""
    binary = (mask == class_id).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0
    # Label 0 is background in connectedComponentsWithStats
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.sum(areas >= min_area_px))


@dataclass(frozen=True)
class MetricResult:
    """Evaluation summary for semantic segmentation on a single frame or confusion matrix."""

    mean_iou: float  # Primary metric: Foreground mIoU over eval_class_ids (default: Pure_Titan, Human)
    all_classes_mean_iou: float  # mIoU including Background (0, 1, 2)
    pixel_accuracy: float
    iou_per_class: dict[str, float]
    confusion_matrix: np.ndarray
    total_valid_pixels: int
    ignored_pixels: int
    pred_titan_count: int = 0
    gt_titan_count: int = 0


def metrics_from_confusion_matrix(
    cm: np.ndarray,
    num_classes: int = 3,
    class_names: dict[int, str] | None = None,
    total_ignored: int = 0,
    eval_class_ids: tuple[int, ...] | None = FOREGROUND_CLASS_IDS,
    pred_titan_count: int = 0,
    gt_titan_count: int = 0,
) -> MetricResult:
    """Compute Foreground mIoU, All-Classes mIoU, class-wise IoU, and pixel accuracy from a confusion matrix.

    Classes without Ground Truth support in the evaluated set (gt_support == 0) are reported as NaN
    and excluded from the mean calculation (Audit B4).
    """
    names = class_names if class_names is not None else CLASS_NAMES_DEFAULT
    target_ids = eval_class_ids if eval_class_ids is not None else tuple(range(num_classes))
    total_valid = int(cm.sum())

    if total_valid == 0:
        return MetricResult(
            mean_iou=0.0,
            all_classes_mean_iou=0.0,
            pixel_accuracy=0.0,
            iou_per_class={names.get(c, f"Class_{c}"): float("nan") for c in range(num_classes)},
            confusion_matrix=cm,
            total_valid_pixels=0,
            ignored_pixels=total_ignored,
            pred_titan_count=pred_titan_count,
            gt_titan_count=gt_titan_count,
        )

    tp = np.diag(cm).astype(np.float64)
    gt_support = cm.sum(axis=1).astype(np.float64)
    pred_support = cm.sum(axis=0).astype(np.float64)
    fn = gt_support - tp
    fp = pred_support - tp
    union = tp + fp + fn

    iou_dict: dict[str, float] = {}
    all_valid_ious: list[float] = []
    fg_valid_ious: list[float] = []

    for c in range(num_classes):
        label = names.get(c, f"Class_{c}")
        if gt_support[c] == 0 or union[c] == 0:
            iou_dict[label] = float("nan")
        else:
            class_iou = float(tp[c] / union[c])
            iou_dict[label] = round(class_iou, 4)
            all_valid_ious.append(class_iou)
            if c in target_ids:
                fg_valid_ious.append(class_iou)

    all_mean_iou = float(np.mean(all_valid_ious)) if all_valid_ious else 0.0
    fg_mean_iou = float(np.mean(fg_valid_ious)) if fg_valid_ious else 0.0
    pixel_acc = float(tp.sum() / total_valid) if total_valid > 0 else 0.0

    return MetricResult(
        mean_iou=round(fg_mean_iou, 4),
        all_classes_mean_iou=round(all_mean_iou, 4),
        pixel_accuracy=round(pixel_acc, 4),
        iou_per_class=iou_dict,
        confusion_matrix=cm,
        total_valid_pixels=total_valid,
        ignored_pixels=total_ignored,
        pred_titan_count=pred_titan_count,
        gt_titan_count=gt_titan_count,
    )


def compute_segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 3,
    ignore_index: int = 255,
    class_names: dict[int, str] | None = None,
    eval_class_ids: tuple[int, ...] | None = FOREGROUND_CLASS_IDS,
    boundary_tolerance_px: int = DEFAULT_BOUNDARY_TOLERANCE_PX,
) -> MetricResult:
    """Compute segmentation metrics using an O(N) confusion matrix with optional contour tolerance."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    if pred.ndim != 2:
        raise ValueError(f"Expected 2D arrays (H, W), got ndim={pred.ndim}")

    invalid_pred_mask = (pred < 0) | (pred >= num_classes)
    if np.any(invalid_pred_mask):
        invalid_ids = np.unique(pred[invalid_pred_mask]).tolist()
        raise ValueError(
            f"Invalid predicted class IDs: {invalid_ids}. "
            f"All predicted pixels must be in range [0, {num_classes - 1}]. "
            "Void (255) and undefined classes cannot be predicted."
        )

    invalid_target_mask = (target != ignore_index) & ((target < 0) | (target >= num_classes))
    if np.any(invalid_target_mask):
        invalid_ids = np.unique(target[invalid_target_mask]).tolist()
        raise ValueError(
            f"Invalid ground truth class IDs: {invalid_ids}. "
            f"Valid classes are [0, {num_classes - 1}] and ignore_index={ignore_index}."
        )

    non_void_mask = target != ignore_index
    valid_mask = non_void_mask
    if boundary_tolerance_px > 0:
        ksize = 2 * boundary_tolerance_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        boundary_band = np.zeros_like(valid_mask, dtype=bool)
        for cid in range(1, num_classes):
            cls_mask = (target == cid).astype(np.uint8)
            if np.any(cls_mask):
                dilated = cv2.dilate(cls_mask, kernel)
                eroded = cv2.erode(cls_mask, kernel)
                boundary_band |= dilated != eroded
        valid_mask = non_void_mask & ~(boundary_band & (pred != target))

    total_valid = int(np.sum(valid_mask))
    total_ignored = int(np.sum(~valid_mask))

    # Count Pure Titan connected components (class_id=1) outside explicit Void regions
    pred_clean = np.where(non_void_mask, pred, 0).astype(np.uint8)
    gt_clean = np.where(non_void_mask, target, 0).astype(np.uint8)
    pred_titans = count_class_instances(pred_clean, class_id=1) if num_classes > 1 else 0
    gt_titans = count_class_instances(gt_clean, class_id=1) if num_classes > 1 else 0

    if total_valid == 0:
        names = class_names if class_names is not None else CLASS_NAMES_DEFAULT
        return MetricResult(
            mean_iou=0.0,
            all_classes_mean_iou=0.0,
            pixel_accuracy=0.0,
            iou_per_class={names.get(c, f"Class_{c}"): float("nan") for c in range(num_classes)},
            confusion_matrix=np.zeros((num_classes, num_classes), dtype=np.int64),
            total_valid_pixels=0,
            ignored_pixels=total_ignored,
            pred_titan_count=pred_titans,
            gt_titan_count=gt_titans,
        )

    valid_pred = pred[valid_mask].astype(np.int64)
    valid_target = target[valid_mask].astype(np.int64)

    category_indices = valid_target * num_classes + valid_pred
    cm = np.bincount(category_indices, minlength=num_classes**2).reshape(num_classes, num_classes)

    return metrics_from_confusion_matrix(
        cm=cm,
        num_classes=num_classes,
        class_names=class_names,
        total_ignored=total_ignored,
        eval_class_ids=eval_class_ids,
        pred_titan_count=pred_titans,
        gt_titan_count=gt_titans,
    )


def compute_pixel_distribution(
    mask: np.ndarray,
    num_classes: int = 3,
    ignore_index: int = 255,
    class_names: dict[int, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute nominal pixel count and percentage distribution per class."""
    names = class_names if class_names is not None else CLASS_NAMES_DEFAULT
    valid_mask = (mask != ignore_index) & (mask >= 0) & (mask < num_classes)
    total_valid = int(np.sum(valid_mask))
    counts = np.bincount(mask[valid_mask].astype(np.int64), minlength=num_classes)

    distribution: dict[str, dict[str, float]] = {}
    for c in range(num_classes):
        label = names.get(c, f"Class_{c}")
        cnt = int(counts[c])
        pct = round(float(cnt / total_valid * 100.0), 2) if total_valid > 0 else 0.0
        distribution[label] = {"count": cnt, "percentage": pct}

    return distribution


@dataclass(frozen=True)
class DatasetEvaluationResult:
    """Aggregated evaluation metrics across multiple frames."""

    mean_iou: float  # Primary: Foreground Micro-mIoU (Pure_Titan, Human)
    all_classes_mean_iou: float  # Micro-mIoU across all 3 classes (Background, Pure_Titan, Human)
    pixel_accuracy: float
    iou_per_class: dict[str, float]
    macro_mean_iou: float
    titan_count_mae: float
    frame_metrics: dict[str, MetricResult]
    pixel_distribution_pred: dict[str, dict[str, float]]
    pixel_distribution_gt: dict[str, dict[str, float]]
    confusion_matrix: np.ndarray
    total_valid_pixels: int
    total_ignored_pixels: int
    num_frames: int

    def to_dict(self) -> dict:
        return {
            "foreground_mean_iou": self.mean_iou,
            "all_classes_mean_iou": self.all_classes_mean_iou,
            "macro_mean_iou": self.macro_mean_iou,
            "pixel_accuracy": self.pixel_accuracy,
            "titan_count_mae": self.titan_count_mae,
            "iou_per_class": {
                k: None if np.isnan(v) else v for k, v in self.iou_per_class.items()
            },
            "num_frames": self.num_frames,
            "total_valid_pixels": self.total_valid_pixels,
            "total_ignored_pixels": self.total_ignored_pixels,
        }


def aggregate_dataset_metrics(
    frame_metrics: dict[str, MetricResult],
    num_classes: int = 3,
    class_names: dict[int, str] | None = None,
    pixel_distribution_pred: dict[str, dict[str, float]] | None = None,
    pixel_distribution_gt: dict[str, dict[str, float]] | None = None,
    eval_class_ids: tuple[int, ...] | None = FOREGROUND_CLASS_IDS,
) -> DatasetEvaluationResult:
    """Aggregate per-frame evaluation results into a single dataset-level report."""
    names = class_names if class_names is not None else CLASS_NAMES_DEFAULT

    if not frame_metrics:
        empty_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        return DatasetEvaluationResult(
            mean_iou=0.0,
            all_classes_mean_iou=0.0,
            pixel_accuracy=0.0,
            iou_per_class={names.get(c, f"Class_{c}"): float("nan") for c in range(num_classes)},
            macro_mean_iou=0.0,
            titan_count_mae=0.0,
            frame_metrics={},
            pixel_distribution_pred={},
            pixel_distribution_gt={},
            confusion_matrix=empty_cm,
            total_valid_pixels=0,
            total_ignored_pixels=0,
            num_frames=0,
        )

    global_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_ignored = 0
    frame_mious: list[float] = []
    count_abs_errors: list[int] = []
    total_pred_titans = 0
    total_gt_titans = 0

    for metric in frame_metrics.values():
        global_cm += metric.confusion_matrix
        total_ignored += metric.ignored_pixels
        frame_mious.append(metric.mean_iou)
        count_abs_errors.append(abs(metric.pred_titan_count - metric.gt_titan_count))
        total_pred_titans += metric.pred_titan_count
        total_gt_titans += metric.gt_titan_count

    global_metrics = metrics_from_confusion_matrix(
        cm=global_cm,
        num_classes=num_classes,
        class_names=names,
        total_ignored=total_ignored,
        eval_class_ids=eval_class_ids,
        pred_titan_count=total_pred_titans,
        gt_titan_count=total_gt_titans,
    )

    macro_mean_iou = round(float(np.mean(frame_mious)), 4) if frame_mious else 0.0
    titan_mae = round(float(np.mean(count_abs_errors)), 4) if count_abs_errors else 0.0

    return DatasetEvaluationResult(
        mean_iou=global_metrics.mean_iou,
        all_classes_mean_iou=global_metrics.all_classes_mean_iou,
        pixel_accuracy=global_metrics.pixel_accuracy,
        iou_per_class=global_metrics.iou_per_class,
        macro_mean_iou=macro_mean_iou,
        titan_count_mae=titan_mae,
        frame_metrics=frame_metrics,
        pixel_distribution_pred=pixel_distribution_pred or {},
        pixel_distribution_gt=pixel_distribution_gt or {},
        confusion_matrix=global_cm,
        total_valid_pixels=global_metrics.total_valid_pixels,
        total_ignored_pixels=total_ignored,
        num_frames=len(frame_metrics),
    )


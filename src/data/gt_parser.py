"""Ground Truth parser and validator for CVAT Segmentation Mask 1.1 format (1080p AoT).

Taxonomy (3 active classes + Void):
- 0: Background (unannotated 'background' in CVAT maps automatically to 0)
- 1: Pure_Titan
- 2: Human
- 255: Void (ignored in evaluation; prohibited in model predictions)
"""

from dataclasses import dataclass
from pathlib import Path
import zipfile
import cv2
import numpy as np


NUM_ACTIVE_CLASSES: int = 3
EXPECTED_HEIGHT: int = 1080
EXPECTED_WIDTH: int = 1920
EXPECTED_SHAPE: tuple[int, int] = (EXPECTED_HEIGHT, EXPECTED_WIDTH)

TAXONOMY_MAP: dict[str, int] = {
    "Background": 0,
    "background": 0,  # Unannotated pixels in CVAT map directly to Background (0)
    "Pure_Titan": 1,
    "Titan": 1,       # Defensive alias for CVAT projects using short label name
    "Human": 2,
    "Void": 255,
}

TAXONOMY_NAMES: dict[int, str] = {
    0: "Background",
    1: "Pure_Titan",
    2: "Human",
    255: "Void",
}

TAXONOMY_COLORS: dict[int, tuple[int, int, int]] = {
    0: (40, 40, 40),      # Background: Dark Gray
    1: (220, 20, 60),     # Pure_Titan: Crimson Red
    2: (0, 191, 255),     # Human: Deep Sky Blue
    255: (255, 0, 255),   # Void: Magenta (Ignored)
}


@dataclass(frozen=True)
class ValidationReport:
    is_valid: bool
    shape: tuple[int, int]
    class_distribution: dict[int, float]
    errors: list[str]


def parse_labelmap(labelmap_content: str) -> dict[tuple[int, int, int], int]:
    """Parse CVAT labelmap.txt into an RGB-tuple -> class_id mapping."""
    rgb_to_id: dict[tuple[int, int, int], int] = {}
    for line in labelmap_content.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        label_name = parts[0].strip()
        color_parts = [int(c) for c in parts[1].split(",")]
        rgb = (color_parts[0], color_parts[1], color_parts[2])
        if label_name not in TAXONOMY_MAP:
            raise ValueError(
                f"Unknown label '{label_name}' in CVAT labelmap.txt. "
                f"Allowed labels: {list(TAXONOMY_MAP.keys())}"
            )
        rgb_to_id[rgb] = TAXONOMY_MAP[label_name]
    return rgb_to_id


def convert_cvat_mask_to_ids(
    mask_bgr: np.ndarray, rgb_to_id: dict[tuple[int, int, int], int]
) -> np.ndarray:
    """Convert an RGB/BGR CVAT mask to a 2D single-channel uint8 array of class IDs."""
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = mask_rgb.shape
    # Default to Background (0) for any unmapped background color unless mapped to Void
    id_mask = np.zeros((h, w), dtype=np.uint8)

    for rgb, class_id in rgb_to_id.items():
        match = np.all(mask_rgb == rgb, axis=-1)
        id_mask[match] = class_id

    return id_mask


def create_color_mask(
    id_mask: np.ndarray,
    color_map: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Render a 2D class ID mask into a 3-channel RGB image."""
    colors = color_map or TAXONOMY_COLORS
    h, w = id_mask.shape
    color_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cid, color in colors.items():
        match = id_mask == cid
        if np.any(match):
            color_rgb[match] = color

    return color_rgb


def create_overlay(
    image_rgb: np.ndarray,
    id_mask: np.ndarray,
    alpha: float = 0.5,
    ignore_void: bool = True,
) -> np.ndarray:
    """Overlay segmentation colors onto the RGB image."""
    color_rgb = create_color_mask(id_mask)
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, color_rgb, alpha, 0)

    if ignore_void:
        void_mask = id_mask == 255
        overlay[void_mask] = image_rgb[void_mask]

    return overlay


def validate_ground_truth_zip(
    zip_path: Path | str,
    expected_shape: tuple[int, int] = EXPECTED_SHAPE,
) -> dict[str, ValidationReport]:
    """Inspect and validate all masks inside a CVAT Segmentation Mask 1.1 zip."""
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_path.name}")

    reports: dict[str, ValidationReport] = {}

    with zipfile.ZipFile(zip_path, "r") as z:
        if "labelmap.txt" not in z.namelist():
            raise ValueError("Invalid format: 'labelmap.txt' missing from zip archive.")

        labelmap_raw = z.read("labelmap.txt").decode("utf-8")
        rgb_to_id = parse_labelmap(labelmap_raw)

        mask_files = [
            f for f in z.namelist() if f.startswith("SegmentationClass/") and f.endswith(".png")
        ]

        for mask_file in mask_files:
            errors: list[str] = []
            data = z.read(mask_file)
            img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            if img_bgr is None:
                errors.append(f"Failed to decode image data for {mask_file}")
                reports[mask_file] = ValidationReport(False, (0, 0), {}, errors)
                continue

            h, w = img_bgr.shape[:2]
            if (h, w) != expected_shape:
                errors.append(f"Resolution mismatch: expected {expected_shape}, got ({h}, {w})")

            id_mask = convert_cvat_mask_to_ids(img_bgr, rgb_to_id)
            total_pixels = h * w
            unique_ids, counts = np.unique(id_mask, return_counts=True)

            dist = {
                int(uid): float(cnt / total_pixels * 100.0)
                for uid, cnt in zip(unique_ids, counts)
            }
            reports[mask_file] = ValidationReport(
                is_valid=len(errors) == 0,
                shape=(h, w),
                class_distribution=dist,
                errors=errors,
            )

    return reports


def load_ground_truth_masks(
    gt_source: Path | str,
    split: str = "all",
    split_stems: list[str] | None = None,
    manifest_path: Path | str | None = None,
) -> dict[str, np.ndarray]:
    """Load decoded ground truth ID masks (2D uint8) from a CVAT ZIP or directory of PNGs."""
    source_path = Path(gt_source)
    if not source_path.exists():
        raise FileNotFoundError(f"Ground truth source not found: {source_path.name}")

    stems = split_stems
    if stems is None and manifest_path is not None:
        from src.data.manifest_schema import DatasetManifest

        manifest = DatasetManifest.load_json(manifest_path)
        if split == "all":
            stems = [f.frame_id for f in manifest.frames]
        else:
            stems = [f.frame_id for f in manifest.frames if f.split == split]

    gt_masks: dict[str, np.ndarray] = {}

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path, "r") as z:
            if "labelmap.txt" not in z.namelist():
                raise ValueError("Invalid format: 'labelmap.txt' missing from zip archive.")

            labelmap_raw = z.read("labelmap.txt").decode("utf-8")
            rgb_to_id = parse_labelmap(labelmap_raw)

            all_mask_files = [
                f for f in z.namelist() if f.startswith("SegmentationClass/") and f.endswith(".png")
            ]

            for mask_file in all_mask_files:
                stem = Path(mask_file).stem
                if stems is not None and stem not in stems:
                    continue

                data = z.read(mask_file)
                img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise ValueError(f"Could not decode image for {mask_file}")
                id_mask = convert_cvat_mask_to_ids(img_bgr, rgb_to_id)
                gt_masks[stem] = id_mask

    elif source_path.is_dir():
        for png_path in sorted(source_path.glob("*.png")):
            stem = png_path.stem
            if stems is not None and stem not in stems:
                continue
            mask = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f"Could not decode PNG mask: {png_path.name}")
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            gt_masks[stem] = mask.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported ground truth source: {source_path.name}")

    if stems is not None:
        for target in stems:
            if target not in gt_masks:
                raise ValueError(f"Target ground truth keyframe missing: {target}")

    return gt_masks


def _validate_and_decode_mask(
    data_bytes: bytes,
    filename: str,
    expected_shape: tuple[int, int] = EXPECTED_SHAPE,
    num_classes: int = NUM_ACTIVE_CLASSES,
) -> np.ndarray:
    """Decode and strictly validate a single predicted mask in memory."""
    mask = cv2.imdecode(np.frombuffer(data_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to decode submission mask: {filename}")

    if mask.ndim == 3:
        if mask.shape[2] in (3, 4) and np.array_equal(mask[:, :, 0], mask[:, :, 1]):
            mask = mask[:, :, 0]
        else:
            raise ValueError(
                f"Invalid format for {filename}: expected single-channel 2D index mask, got 3D shape {mask.shape}. "
                f"Submit 8-bit index masks with integer values [0, {num_classes - 1}]."
            )

    if mask.shape != expected_shape:
        raise ValueError(
            f"Resolution mismatch for {filename}: expected {expected_shape}, got {mask.shape}"
        )

    min_val, max_val = int(mask.min()), int(mask.max())
    if min_val < 0 or max_val >= num_classes:
        invalid_vals = np.unique(mask[(mask < 0) | (mask >= num_classes)])
        raise ValueError(
            f"Invalid class IDs in {filename}: found values {invalid_vals.tolist()}. "
            f"All predicted pixels must be in range [0, {num_classes - 1}]. "
            "Models cannot predict Void (255) or undefined classes."
        )

    return mask.astype(np.uint8)


def iter_submission_masks(
    submission_path: Path | str,
    target_stems: list[str],
    expected_shape: tuple[int, int] = EXPECTED_SHAPE,
    num_classes: int = NUM_ACTIVE_CLASSES,
):
    """Lazy generator yielding (target_stem, 2D uint8 mask) one frame at a time (O(1) RAM)."""
    sub_path = Path(submission_path)
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission path not found: {sub_path.name}")

    if sub_path.is_file() and sub_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(sub_path, "r") as z:
            available_files = [f for f in z.namelist() if f.lower().endswith(".png")]
            stem_to_file = {Path(f).stem: f for f in available_files}
            for target in target_stems:
                matched_file = stem_to_file.get(target)
                if matched_file is None:
                    raise FileNotFoundError(
                        f"Missing prediction for keyframe '{target}' in submission zip."
                    )
                data = z.read(matched_file)
                mask = _validate_and_decode_mask(data, matched_file, expected_shape, num_classes)
                yield target, mask

    elif sub_path.is_dir():
        available_files = list(sub_path.glob("**/*.png"))
        stem_to_path = {f.stem: f for f in available_files}
        for target in target_stems:
            matched_path = stem_to_path.get(target)
            if matched_path is None:
                raise FileNotFoundError(
                    f"Missing prediction for keyframe '{target}' in submission directory."
                )
            with open(matched_path, "rb") as f:
                data = f.read()
            mask = _validate_and_decode_mask(data, matched_path.name, expected_shape, num_classes)
            yield target, mask
    else:
        raise ValueError(f"Submission path must be a .zip file or a directory: {sub_path.name}")


def load_submission_masks(
    submission_path: Path | str,
    target_stems: list[str],
    expected_shape: tuple[int, int] = EXPECTED_SHAPE,
    num_classes: int = NUM_ACTIVE_CLASSES,
) -> dict[str, np.ndarray]:
    """Load and validate student predicted masks into a dictionary."""
    return dict(iter_submission_masks(submission_path, target_stems, expected_shape, num_classes))


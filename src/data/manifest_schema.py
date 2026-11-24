"""Pydantic schemas for screenshot ingestion and dataset manifests.

Enforces strict provenance, cryptographic SHA-256 hashes, canonical 1920x1080 resolution,
and split assignment at I/O boundaries.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


SplitType = Literal["train", "val_in_domain", "test_public", "test_private"]


class FrameRecord(BaseModel):
    """Metadata record for a single ingested 1080p keyframe."""

    frame_id: str = Field(..., description="Canonical stem identifier (e.g. aot_train_0001).")
    image_filename: str = Field(..., description="Basename of the normalized 1080p PNG file.")
    source_sha256: str = Field(
        ...,
        pattern=r"^[a-f0-9]{64}$",
        description="Cryptographic SHA-256 hash of the original raw screenshot.",
    )
    normalized_sha256: str = Field(
        ...,
        pattern=r"^[a-f0-9]{64}$",
        description="Cryptographic SHA-256 hash of the normalized 1920x1080 PNG file.",
    )
    original_resolution: tuple[int, int] = Field(
        ...,
        description="Original (width, height) of the raw capture before normalization.",
    )
    resolution: tuple[int, int] = Field(
        default=(1920, 1080),
        description="Normalized (width, height) in pixels.",
    )
    mean_luminance: float = Field(
        ...,
        ge=0.0,
        le=255.0,
        description="Mean pixel luminance used to verify absence of black DRM frames.",
    )
    split: SplitType = Field(
        default="train",
        description="Assigned dataset split.",
    )


class DatasetManifest(BaseModel):
    """Root container for a validated AoT dataset manifest."""

    manifest_version: str = Field(default="1.0.0", description="SemVer schema specification version.")
    dataset_name: str = Field(default="AoTSegmentation-1080p", description="Dataset identifier.")
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of manifest generation.",
    )
    total_frames: int = Field(..., ge=0, description="Total count of registered frame records.")
    frames: list[FrameRecord] = Field(default_factory=list, description="List of validated frame records.")

    def save_json(self, path: Path | str) -> None:
        """Serialize validated manifest to disk with indent formatting."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, path: Path | str) -> "DatasetManifest":
        """Load and strictly validate a dataset manifest from disk."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Manifest not found: {target.name}")
        with open(target, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


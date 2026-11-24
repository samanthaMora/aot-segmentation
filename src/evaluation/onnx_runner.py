"""ONNX model execution harness, contract validator, and latency benchmarking (1080p AoT).

Contract Specification:
- Input: 'image', shape [1, 3, 1080, 1920], dtype uint8, range [0, 255].
- Output: 'mask', shape [1, 1080, 1920], dtype uint8 or int64, classes {0, 1, 2}.
- Max model size: 250 MB.
- Opset version: 17 to 20.
- Prohibited class: 255 (Void) cannot be predicted by models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort


MAX_MODEL_SIZE_BYTES: int = 250 * 1024 * 1024  # 250 MB
ALLOWED_OPSETS: tuple[int, ...] = (17, 18, 19, 20)
EXPECTED_HEIGHT: int = 1080
EXPECTED_WIDTH: int = 1920
EXPECTED_INPUT_NAME: str = "image"
EXPECTED_INPUT_SHAPE: tuple[int, ...] = (1, 3, EXPECTED_HEIGHT, EXPECTED_WIDTH)
EXPECTED_INPUT_TYPE: str = "tensor(uint8)"
EXPECTED_OUTPUT_NAME: str = "mask"
EXPECTED_OUTPUT_SHAPE: tuple[int, ...] = (1, EXPECTED_HEIGHT, EXPECTED_WIDTH)
ALLOWED_OUTPUT_TYPES: tuple[str, ...] = ("tensor(uint8)", "tensor(int64)")
NUM_CLASSES: int = 3


class OnnxContractViolation(ValueError):
    """Raised when an ONNX model violates the official submission contract."""


@dataclass(frozen=True)
class OnnxContractReport:
    """Report summarizing ONNX contract compliance."""

    is_valid: bool
    file_size_bytes: int
    opset_version: int | None
    input_name: str
    input_shape: tuple[int | str, ...]
    input_type: str
    output_name: str
    output_shape: tuple[int | str, ...]
    output_type: str
    errors: list[str] = field(default_factory=list)

    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)


@dataclass(frozen=True)
class LatencyBenchmarkResult:
    """Benchmark results for ONNX inference latency."""

    p50_ms: float
    p90_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    warmup_runs: int
    timed_runs: int
    durations_ms: list[float]


def validate_onnx_contract(
    model_path: Path | str,
    raise_on_error: bool = False,
) -> OnnxContractReport:
    """Validate an ONNX model against the official AoT 1080p competition submission contract."""
    path = Path(model_path)
    errors: list[str] = []

    if not path.is_file():
        msg = f"Model file not found: {path.name}"
        if raise_on_error:
            raise OnnxContractViolation(msg)
        return OnnxContractReport(
            is_valid=False,
            file_size_bytes=0,
            opset_version=None,
            input_name="",
            input_shape=(),
            input_type="",
            output_name="",
            output_shape=(),
            output_type="",
            errors=[msg],
        )

    file_size = path.stat().st_size
    if file_size > MAX_MODEL_SIZE_BYTES:
        errors.append(
            f"File size {file_size / (1024 * 1024):.2f} MB exceeds limit of "
            f"{MAX_MODEL_SIZE_BYTES / (1024 * 1024):.2f} MB"
        )

    opset_ver: int | None = None
    input_name, input_shape, input_type = "", (), ""
    output_name, output_shape, output_type = "", (), ""

    try:
        model_proto = onnx.load(str(path), load_external_data=False)
        for op in model_proto.opset_import:
            if op.domain in ("", "ai.onnx"):
                opset_ver = op.version
                break
        if opset_ver is None or opset_ver not in ALLOWED_OPSETS:
            errors.append(
                f"Opset version {opset_ver} is not allowed. Must be one of {ALLOWED_OPSETS}."
            )

        graph_inputs = list(model_proto.graph.input)
        if len(graph_inputs) != 1:
            errors.append(f"Model must have exactly 1 input, found {len(graph_inputs)}.")
        else:
            ginp = graph_inputs[0]
            input_name = ginp.name
            dims = [
                d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in ginp.type.tensor_type.shape.dim
            ]
            input_shape = tuple(dims)
            elem_type = ginp.type.tensor_type.elem_type
            type_str = f"tensor({TensorProto.DataType.Name(elem_type).lower()})"
            input_type = type_str
            if input_name != EXPECTED_INPUT_NAME:
                errors.append(f"Input name must be '{EXPECTED_INPUT_NAME}', got '{input_name}'.")
            if input_shape != EXPECTED_INPUT_SHAPE:
                errors.append(f"Input shape must be {EXPECTED_INPUT_SHAPE}, got {input_shape}.")
            if input_type != EXPECTED_INPUT_TYPE:
                errors.append(f"Input type must be '{EXPECTED_INPUT_TYPE}', got '{input_type}'.")

        graph_outputs = list(model_proto.graph.output)
        if len(graph_outputs) != 1:
            errors.append(f"Model must have exactly 1 output, found {len(graph_outputs)}.")
        else:
            gout = graph_outputs[0]
            output_name = gout.name
            dims = [
                d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in gout.type.tensor_type.shape.dim
            ]
            output_shape = tuple(dims)
            elem_type = gout.type.tensor_type.elem_type
            type_str = f"tensor({TensorProto.DataType.Name(elem_type).lower()})"
            output_type = type_str
            if output_name != EXPECTED_OUTPUT_NAME:
                errors.append(f"Output name must be '{EXPECTED_OUTPUT_NAME}', got '{output_name}'.")
            if output_shape != EXPECTED_OUTPUT_SHAPE:
                errors.append(f"Output shape must be {EXPECTED_OUTPUT_SHAPE}, got {output_shape}.")
            if output_type not in ALLOWED_OUTPUT_TYPES:
                errors.append(
                    f"Output type must be one of {ALLOWED_OUTPUT_TYPES}, got '{output_type}'."
                )

    except Exception as exc:
        errors.append(f"Failed to parse ONNX protobuf: {exc}")

    if not errors:
        try:
            _ = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as exc:
            errors.append(f"InferenceSession creation failed: {exc}")

    is_valid = len(errors) == 0
    if not is_valid and raise_on_error:
        raise OnnxContractViolation("\n".join(errors))

    return OnnxContractReport(
        is_valid=is_valid,
        file_size_bytes=file_size,
        opset_version=opset_ver,
        input_name=input_name,
        input_shape=input_shape,
        input_type=input_type,
        output_name=output_name,
        output_shape=output_shape,
        output_type=output_type,
        errors=errors,
    )


class OnnxModelRunner:
    """Execution engine for validated ONNX submission models."""

    def __init__(
        self,
        model_path: Path | str,
        providers: Sequence[str] | None = None,
        validate_contract: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        if validate_contract:
            self.report = validate_onnx_contract(self.model_path, raise_on_error=True)
        else:
            self.report = validate_onnx_contract(self.model_path, raise_on_error=False)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        active_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=active_providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run single-frame inference on 1080p RGB input.

        Args:
            image: 3D uint8 array [1080, 1920, 3] (RGB) or 4D uint8 array [1, 3, 1080, 1920].

        Returns:
            2D uint8 mask [1080, 1920] with class values in {0, 1, 2}.
        """
        if image.ndim == 3:
            if image.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
                raise ValueError(
                    f"Expected image shape ({EXPECTED_HEIGHT}, {EXPECTED_WIDTH}, 3), got {image.shape}"
                )
            tensor = np.ascontiguousarray(
                image.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.uint8
            )
        elif image.ndim == 4:
            if image.shape != EXPECTED_INPUT_SHAPE:
                raise ValueError(
                    f"Expected tensor shape {EXPECTED_INPUT_SHAPE}, got {image.shape}"
                )
            tensor = np.ascontiguousarray(image, dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")

        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        raw_mask = outputs[0]

        if raw_mask.ndim == 3 and raw_mask.shape[0] == 1:
            mask_2d = raw_mask[0]
        elif raw_mask.ndim == 2:
            mask_2d = raw_mask
        else:
            raise ValueError(f"Unexpected output mask shape from model: {raw_mask.shape}")

        min_val = int(mask_2d.min())
        max_val = int(mask_2d.max())
        if min_val < 0 or max_val >= NUM_CLASSES:
            if 255 in mask_2d:
                raise OnnxContractViolation(
                    "Model predicted class 255 (Void). Void is reserved exclusively for "
                    "human ground-truth masking and is strictly prohibited in predictions."
                )
            raise OnnxContractViolation(
                f"Model predicted out-of-range class ID [{min_val}, {max_val}]. "
                f"Must be within [0, {NUM_CLASSES - 1}]."
            )

        return mask_2d.astype(np.uint8)

    def benchmark_latency(
        self,
        warmup_runs: int = 20,
        timed_runs: int = 30,
    ) -> LatencyBenchmarkResult:
        """Execute the official latency benchmark on hardware using dummy 1080p input."""
        dummy_input = np.zeros(EXPECTED_INPUT_SHAPE, dtype=np.uint8)
        feed_dict = {self.input_name: dummy_input}
        output_names = [self.output_name]

        for _ in range(warmup_runs):
            self.session.run(output_names, feed_dict)

        durations: list[float] = []
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            self.session.run(output_names, feed_dict)
            durations.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(durations, dtype=np.float64)
        return LatencyBenchmarkResult(
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            mean_ms=float(arr.mean()),
            std_ms=float(arr.std()),
            min_ms=float(arr.min()),
            max_ms=float(arr.max()),
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            durations_ms=durations,
        )


def create_dummy_onnx_model(
    output_path: Path | str,
    opset: int = 17,
    ir_version: int = 10,
    class_id: int = 0,
) -> Path:
    """Generate a minimal valid ONNX model adhering strictly to the 1080p AoT submission contract."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image_info = helper.make_tensor_value_info(
        EXPECTED_INPUT_NAME, TensorProto.UINT8, list(EXPECTED_INPUT_SHAPE)
    )
    mask_info = helper.make_tensor_value_info(
        EXPECTED_OUTPUT_NAME, TensorProto.UINT8, list(EXPECTED_OUTPUT_SHAPE)
    )

    axes_node = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    starts_node = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    ends_node = helper.make_tensor("ends", TensorProto.INT64, [1], [1])
    sq_axes = helper.make_tensor("sq_axes", TensorProto.INT64, [1], [1])
    class_val = helper.make_tensor(
        "target_class", TensorProto.UINT8, [1], [max(1, min(class_id + 1, NUM_CLASSES))]
    )

    slice_node = helper.make_node(
        "Slice", [EXPECTED_INPUT_NAME, "starts", "ends", "axes"], ["sliced"]
    )
    sq_node = helper.make_node("Squeeze", ["sliced", "sq_axes"], ["squeezed"])
    mod_node = helper.make_node("Mod", ["squeezed", "target_class"], [EXPECTED_OUTPUT_NAME], fmod=0)

    graph = helper.make_graph(
        [slice_node, sq_node, mod_node],
        "dummy_aot_submission_model",
        [image_info],
        [mask_info],
        initializer=[axes_node, starts_node, ends_node, sq_axes, class_val],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=ir_version,
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


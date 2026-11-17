"""Offline OpenVINO IR inspection through the native arm64 inspector."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any


class NCS2ModelInspectionError(RuntimeError):
    """The inspector could not read or describe an IR model."""


@dataclass(frozen=True)
class NCS2TensorInfo:
    index: int
    name: str
    dtype: str
    is_static: bool
    rank: int | None
    shape: tuple[int, ...] | None
    elements: int | None
    element_limit_ok: bool


@dataclass(frozen=True)
class NCS2Compatibility:
    compatible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class NCS2ModelInspection:
    xml: Path
    openvino_version: str
    inputs: tuple[NCS2TensorInfo, ...]
    outputs: tuple[NCS2TensorInfo, ...]
    schema_1: NCS2Compatibility
    schema_2: NCS2Compatibility
    hardware_accessed: bool = False

    @property
    def supported_schemas(self) -> tuple[int, ...]:
        return tuple(schema for schema, value in ((1, self.schema_1), (2, self.schema_2))
                     if value.compatible)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "NCS2ModelInspection":
        if (type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1
                or payload.get("type") != "model_inspection"
                or payload.get("hardware_accessed") is not False):
            raise NCS2ModelInspectionError("Invalid model inspection identity")
        model = payload.get("model")
        matrix = payload.get("compatibility")
        if not isinstance(model, dict) or not isinstance(matrix, dict):
            raise NCS2ModelInspectionError("Invalid model inspection payload")

        def tensors(value: Any, label: str) -> tuple[NCS2TensorInfo, ...]:
            if not isinstance(value, list):
                raise NCS2ModelInspectionError(f"Invalid {label} tensor list")
            result = []
            for item in value:
                if not isinstance(item, dict):
                    raise NCS2ModelInspectionError(f"Invalid {label} tensor")
                shape = item.get("shape")
                parsed_shape = None if shape is None else tuple(shape)
                if parsed_shape is not None and any(type(dim) is not int or dim <= 0 for dim in parsed_shape):
                    raise NCS2ModelInspectionError(f"Invalid {label} shape")
                rank = item.get("rank")
                if rank is not None and (type(rank) is not int or rank < 0):
                    raise NCS2ModelInspectionError(f"Invalid {label} rank")
                if parsed_shape is not None and rank is not None and len(parsed_shape) != rank:
                    raise NCS2ModelInspectionError(f"{label} shape and rank disagree")
                elements = item.get("elements")
                if elements is not None and (type(elements) is not int or elements < 0):
                    raise NCS2ModelInspectionError(f"Invalid {label} element count")
                if (type(item.get("index")) is not int or not isinstance(item.get("name"), str)
                        or not isinstance(item.get("dtype"), str) or type(item.get("static")) is not bool
                        or type(item.get("element_limit_ok")) is not bool):
                    raise NCS2ModelInspectionError(f"Invalid {label} tensor metadata")
                result.append(NCS2TensorInfo(item["index"], item["name"], item["dtype"], item["static"],
                                             rank, parsed_shape, elements, item["element_limit_ok"]))
            if [item.index for item in result] != list(range(len(result))):
                raise NCS2ModelInspectionError(f"{label} tensor indexes are not ordered")
            return tuple(result)

        def compatibility(value: Any, label: str) -> NCS2Compatibility:
            if not isinstance(value, dict) or type(value.get("compatible")) is not bool:
                raise NCS2ModelInspectionError(f"Invalid {label} compatibility")
            reasons = value.get("reasons")
            if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
                raise NCS2ModelInspectionError(f"Invalid {label} compatibility reasons")
            return NCS2Compatibility(value["compatible"], tuple(reasons))

        xml = model.get("xml")
        if not isinstance(xml, str) or not isinstance(payload.get("openvino_version"), str):
            raise NCS2ModelInspectionError("Invalid model inspection metadata")
        inputs = tensors(model.get("inputs"), "input")
        outputs = tensors(model.get("outputs"), "output")
        for key, expected in (("input_count", len(inputs)), ("output_count", len(outputs))):
            if type(model.get(key)) is not int or model.get(key) != expected:
                raise NCS2ModelInspectionError(f"Invalid {key}")
        return cls(Path(xml), payload["openvino_version"], inputs, outputs,
                   compatibility(matrix.get("schema_1"), "schema 1"),
                   compatibility(matrix.get("schema_2"), "schema 2"))


class NCS2ModelInspector:
    """Call the native offline inspector without opening the NCS2."""

    def __init__(self, executable: Path):
        self.executable = Path(executable).absolute()

    def inspect(self, xml: Path, *, timeout: float = 30) -> NCS2ModelInspection:
        if type(timeout) not in (int, float) or not 0 < timeout <= 300:
            raise NCS2ModelInspectionError("Inspection timeout must be in (0,300]")
        xml = Path(xml).absolute()
        try:
            completed = subprocess.run([str(self.executable), "--model", str(xml)],
                                       check=False, capture_output=True, text=True,
                                       timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise NCS2ModelInspectionError(f"Model inspector could not run: {error}") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "Native inspector failed"
            raise NCS2ModelInspectionError(f"{detail} (exit status {completed.returncode})")
        try:
            payload = json.loads(completed.stdout)
        except (ValueError, UnicodeError) as error:
            raise NCS2ModelInspectionError("Model inspector returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise NCS2ModelInspectionError("Model inspector returned a non-object")
        return NCS2ModelInspection.from_payload(payload)


def inspect_model(xml: Path, executable: Path | None = None, *, timeout: float = 30) -> NCS2ModelInspection:
    """Inspect an IR XML/BIN pair; no device lease or USB access is used."""
    if executable is None:
        configured = os.environ.get("NCS2_MODEL_INSPECTOR")
        candidates = ([Path(configured)] if configured else []) + [
            Path.cwd() / "build-arm64/ncs2_model_inspect",
            Path(__file__).resolve().parent / "build-arm64/ncs2_model_inspect",
        ]
        executable = (Path(configured) if configured else
                      next((path for path in candidates if path.is_file()), candidates[-1]))
    return NCS2ModelInspector(executable).inspect(xml, timeout=timeout)

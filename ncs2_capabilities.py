"""The public, device-independent NeuralStick Bridge capability matrix."""
from __future__ import annotations

import argparse
import json


CAPABILITY_MATRIX = {
    "schema_version": 1,
    "project": "NeuralStick Bridge",
    "host": {
        "os": "macOS",
        "architecture": "arm64",
        "native_only": True,
        "rosetta": False,
    },
    "device": {
        "family": "Intel Neural Compute Stick 2",
        "accelerator": "MyriadX / MYRIAD",
        "camera_required": False,
        "concurrent_inferences_per_runtime": 1,
    },
    "models": {
        "source": "OpenVINO IR XML + BIN",
        "static_shape_only": True,
        "dtype": ["f32"],
        "rank": {"min": 1, "max": 8},
        "tensor_elements": {"per_tensor_max": 16384, "aggregate_input_max": 16384,
                             "aggregate_output_max": 16384},
        "schema_1": {"inputs": 1, "outputs": 1},
        "schema_2": {"inputs": {"min": 1, "max": 32},
                     "outputs": {"min": 1, "max": 32}},
        "dynamic_shapes": False,
    },
    "protocol": {
        "transport": "JSON Lines over stdin/stdout",
        "frame_bytes_max": 1048576,
        "worker_schema_versions": [1, 2],
        "inference_timer": "monotonic duration from start_async through wait_for",
        "fatal_inference_recovery": "reap worker and retry after resubmission",
        "cpu_fallback": False,
    },
    "sdk": {
        "languages": ["Python asyncio", "Swift async/await"],
        "queue": {"fifo": True, "waiting_tasks_max": 32, "running_tasks_max": 1},
        "queued_cancellation": True,
        "running_cancellation": False,
        "model_registration": "in-process",
        "result_lifecycle": "explicit release, optional retention limits",
    },
    "unsupported": [
        "Linux, VM, container, Rosetta, and x86_64 runtime",
        "dynamic-shape or non-f32 worker inputs/outputs",
        "more than one concurrent inference in one runtime",
        "camera integration",
    ],
}


def capabilities() -> dict:
    """Return a detached JSON-compatible copy of the current matrix."""
    return json.loads(json.dumps(CAPABILITY_MATRIX))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    args = parser.parse_args(argv)
    matrix = capabilities()
    if args.json:
        print(json.dumps(matrix, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    print("NeuralStick Bridge capability matrix")
    print("host: macOS arm64 (native only)")
    print("device: Intel Neural Compute Stick 2 / MyriadX MYRIAD")
    print("models: static f32 IR; rank 1..8; 16384 elements per tensor and per aggregate direction")
    print("schemas: schema 1 = 1 input/1 output; schema 2 = 1..32 inputs/outputs")
    print("queue: FIFO, 1 running task, 32 waiting tasks")
    print("transport: bounded JSON Lines, 1 MiB maximum frame")
    print("fallback: CPU fallback disabled; dynamic shapes unsupported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

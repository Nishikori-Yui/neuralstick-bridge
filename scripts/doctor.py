#!/usr/bin/env python3
"""Check the local toolchain; USB checks run only with --hardware."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def check(name: str, ok: bool, detail: str, *, required: bool = True) -> dict:
    return {"name": name, "ok": bool(ok), "required": required, "detail": detail}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_capture(*args: str) -> str | None:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def binary_check(name: str, path: Path | None) -> dict:
    architecture = run_capture("lipo", "-archs", str(path)) if path and path.is_file() else None
    return check(name, architecture == "arm64", f"{path or 'not found'}; arch={architecture or 'unavailable'}")


def firmware_checks(root: Path, lock: dict) -> list[dict]:
    results = []
    cache = root / "firmware-cache"
    for item in lock["firmware"]:
        output = cache / item["output"]
        ok = output.is_file()
        detail = str(output.relative_to(root)) if ok else f"missing {output.relative_to(root)}"
        if ok:
            digest = sha256(output)
            detail += f" sha256={digest}"
            ok = digest == item["output_sha256"]
            if not ok:
                detail += " (mismatch)"
        results.append(check(f"firmware:{item['output']}", ok, detail))
    return results


def diagnose(root: Path = ROOT, *, hardware: bool = False) -> dict:
    lock_path = root / "toolchain.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    ov = lock["openvino"]
    source = root / ov["directory"]
    runtime = source / "bin/arm64/Release"
    build = root / "build-arm64"
    results: list[dict] = []

    results.append(check("host.os", platform.system() == "Darwin", platform.system()))
    results.append(check("host.architecture", platform.machine() == "arm64", platform.machine()))
    translated = run_capture("/usr/sbin/sysctl", "-n", "sysctl.proc_translated")
    results.append(check("host.native", translated != "1" and platform.machine() == "arm64",
                         "Rosetta" if translated == "1" else platform.machine()))
    results.append(check("python", sys.version_info >= (3, 11),
                         f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))
    for command in ("xcrun", "clang++", "cmake", "pkg-config"):
        path = shutil.which(command)
        results.append(check(f"command:{command}", path is not None, path or "not found"))
    clang = run_capture("xcrun", "--find", "clang++")
    results.append(check("xcode.clang", clang is not None, clang or "select Xcode or Command Line Tools"))

    source_revision = run_capture("git", "-C", str(source), "rev-parse", "HEAD") if (source / ".git").is_dir() else None
    results.append(check("openvino.source", source_revision == ov["commit"],
                         source_revision or f"missing {source.relative_to(root)}"))
    for directory, revision in ov.get("submodules", {}).items():
        actual = run_capture("git", "-C", str(source / directory), "rev-parse", "HEAD")
        results.append(check(f"submodule:{directory}", actual == revision, actual or "missing"))
    results.append(check("openvino.plugins.xml", (runtime / "plugins.xml").is_file(),
                         str((runtime / "plugins.xml").relative_to(root))))
    for name in ("libopenvino.2022.3.2.dylib", "libopenvino_intel_myriad_plugin.so"):
        path = runtime / name
        results.append(binary_check(f"runtime:{name}", path))

    libusb = os.environ.get("NCS2_LIBUSB")
    libusb_candidates = [Path(libusb)] if libusb else []
    libusb_candidates += [] if libusb else [
        root / "external/libusb-1.0.0.dylib",
        Path("/opt/homebrew/opt/libusb/lib/libusb-1.0.0.dylib"),
        Path("/usr/local/opt/libusb/lib/libusb-1.0.0.dylib"),
    ]
    libusb_path = next((path for path in libusb_candidates if path.is_file()), None)
    results.append(binary_check("libusb.arm64", libusb_path))

    for name in ("ncs2_openvino_probe", "ncs2_model_inspect", "ncs2_worker", "ncs2_multi_worker"):
        path = build / name
        results.append(check(f"build:{name}", path.is_file() and os.access(path, os.X_OK),
                             str(path.relative_to(root))))
        results.append(binary_check(f"architecture:{name}", path))
    results.extend(firmware_checks(root, lock))
    results.append(check("device.lock", True,
                         os.environ.get("NCS2_DEVICE_LOCK",
                                       f"/tmp/neuralstick-bridge-device-{os.getuid()}.lock"),
                         required=False))

    if hardware:
        commands = {
            "hardware.usb": [sys.executable, str(root / "read_ncs2_usb.py")],
            "hardware.myriad": ["/bin/zsh", str(root / "scripts/run_probe.sh"), "list"],
        }
        for name, command in commands.items():
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=30)
                detail = (response.stdout + response.stderr)[-8192:].strip()
                results.append(check(name, response.returncode == 0, detail or f"exit={response.returncode}"))
            except (OSError, subprocess.TimeoutExpired) as error:
                results.append(check(name, False, str(error)))

    failed_required = [item for item in results if item["required"] and not item["ok"]]
    return {
        "schema_version": 1,
        "project": "NeuralStick Bridge",
        "hardware_accessed": hardware,
        "ok": not failed_required,
        "checks": results,
        "failed": [item["name"] for item in failed_required],
    }


def render(report: dict) -> str:
    lines = [f"NeuralStick Bridge doctor: {'PASS' if report['ok'] else 'NEEDS ATTENTION'}"]
    for item in report["checks"]:
        marker = "ok" if item["ok"] else "FAIL"
        suffix = "" if item["required"] else " (optional)"
        lines.append(f"[{marker}] {item['name']}: {item['detail']}{suffix}")
    if not report["ok"]:
        lines.append("Run ./ncs2 fetch --offline or ./ncs2 build to prepare missing toolchain files.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit one JSON report")
    parser.add_argument("--hardware", action="store_true", help="also open USB and query MYRIAD (no inference)")
    args = parser.parse_args(argv)
    try:
        report = diagnose(hardware=args.hardware)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as error:
        print(f"doctor: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True) if args.json else render(report))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())

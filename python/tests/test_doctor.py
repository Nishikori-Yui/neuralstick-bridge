"""Doctor checks are read-only and report missing build state precisely."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/doctor.py"
SPEC = importlib.util.spec_from_file_location("doctor", SCRIPT)
doctor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doctor)


class DoctorTests(unittest.TestCase):
    def test_clean_checkout_reports_missing_local_runtime_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = {
                "openvino": {"commit": "abc", "directory": "vendor/openvino-2022.3"},
                "firmware": [{"output": "usb/test.mvcmd", "output_sha256": "0" * 64}],
            }
            (root / "toolchain.lock.json").write_text(json.dumps(lock), encoding="utf-8")
            report = doctor.diagnose(root)
            self.assertFalse(report["ok"])
            self.assertIn("openvino.source", report["failed"])
            self.assertIn("build:ncs2_worker", report["failed"])
            self.assertFalse(report["hardware_accessed"])
            self.assertIn("NEEDS ATTENTION", doctor.render(report))

    def test_firmware_checksum_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"firmware"
            import hashlib
            digest = hashlib.sha256(data).hexdigest()
            item = {"output": "usb/test.mvcmd", "output_sha256": digest}
            (root / "firmware-cache/usb").mkdir(parents=True)
            (root / "firmware-cache/usb/test.mvcmd").write_bytes(data)
            (root / "toolchain.lock.json").write_text(json.dumps({
                "openvino": {"commit": "abc", "directory": "vendor/openvino-2022.3"},
                "firmware": [item],
            }), encoding="utf-8")
            report = doctor.diagnose(root)
            firmware = next(item for item in report["checks"] if item["name"] == "firmware:usb/test.mvcmd")
            self.assertTrue(firmware["ok"])

    def test_binary_check_rejects_universal_and_x86(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.dylib"
            path.write_bytes(b"synthetic")
            for arch, expected in [("arm64", True), ("arm64 x86_64", False), ("x86_64", False), (None, False)]:
                with mock.patch.object(doctor, "run_capture", return_value=arch):
                    self.assertEqual(doctor.binary_check("binary", path)["ok"], expected)

    def test_json_launcher_stdout_is_one_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "doctor.md"
            process = subprocess.run(["/bin/zsh", str(ROOT / "ncs2"), "doctor", "--json"],
                                     capture_output=True, text=True, timeout=30,
                                     env={**os.environ, "NCS2_LOG_FILE": str(log)})
            self.assertIn(process.returncode, (0, 2), process.stderr)
            report = json.loads(process.stdout)
            self.assertFalse(report["hardware_accessed"])
            self.assertTrue(log.is_file())
            self.assertIn("[recorded]", process.stderr)


if __name__ == "__main__":
    unittest.main()

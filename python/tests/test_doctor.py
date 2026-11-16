"""Doctor checks are read-only and report missing build state precisely."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()

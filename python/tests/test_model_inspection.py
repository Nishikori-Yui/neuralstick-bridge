"""Offline model-inspector API and capability-matrix contract tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ncs2_model import NCS2ModelInspectionError, NCS2ModelInspector
from ncs2_capabilities import capabilities


class ModelInspectionTests(unittest.TestCase):
    def payload(self):
        return {
            "schema_version": 1,
            "type": "model_inspection",
            "hardware_accessed": False,
            "openvino_version": "2022.3.2",
            "model": {
                "xml": "/models/relu.xml",
                "inputs": [{"index": 0, "name": "input", "dtype": "f32", "static": True,
                            "rank": 2, "shape": [1, 4], "elements": 4, "element_limit_ok": True}],
                "outputs": [{"index": 0, "name": "output", "dtype": "f32", "static": True,
                             "rank": 2, "shape": [1, 4], "elements": 4, "element_limit_ok": True}],
                "input_count": 1,
                "output_count": 1,
            },
            "compatibility": {
                "schema_1": {"compatible": True, "reasons": []},
                "schema_2": {"compatible": True, "reasons": []},
            },
        }

    def run_inspector(self, directory, stdout, exit_code=0):
        path = Path(directory) / "inspector"
        literal = repr(stdout)
        path.write_text("#!/usr/bin/env python3\nimport sys\nsys.stdout.write(" + literal + ")\nsys.exit(" + str(exit_code) + ")\n")
        path.chmod(0o755)
        return path

    def test_inspects_static_contract_without_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = self.run_inspector(directory, json.dumps(self.payload()))
            result = NCS2ModelInspector(executable).inspect(Path("models/relu.xml"))
        self.assertEqual(result.supported_schemas, (1, 2))
        self.assertEqual(result.inputs[0].shape, (1, 4))
        self.assertFalse(result.hardware_accessed)

    def test_rejects_untrusted_or_inconsistent_inspector_payload(self):
        cases = []
        payload = self.payload()
        payload["hardware_accessed"] = True
        cases.append(payload)
        payload = self.payload()
        payload["model"]["input_count"] = 2
        cases.append(payload)
        payload = self.payload()
        payload["model"]["inputs"][0]["shape"] = [True, 4]
        cases.append(payload)
        payload = self.payload()
        payload["model"]["inputs"][0]["rank"] = 3
        cases.append(payload)
        with tempfile.TemporaryDirectory() as directory:
            for index, payload in enumerate(cases):
                with self.subTest(index=index):
                    executable = self.run_inspector(directory, json.dumps(payload))
                    with self.assertRaises(NCS2ModelInspectionError):
                        NCS2ModelInspector(executable).inspect(Path("models/relu.xml"))

    def test_scalar_tensor_is_reported_as_unsupported_not_malformed(self):
        payload = self.payload()
        for direction in ("inputs", "outputs"):
            payload["model"][direction][0].update(rank=0, shape=[], elements=None,
                                                   element_limit_ok=False)
        payload["compatibility"]["schema_1"] = {
            "compatible": False,
            "reasons": ["rank must be 1..8"],
        }
        payload["compatibility"]["schema_2"] = {
            "compatible": False,
            "reasons": ["rank must be 1..8"],
        }
        with tempfile.TemporaryDirectory() as directory:
            executable = self.run_inspector(directory, json.dumps(payload))
            result = NCS2ModelInspector(executable).inspect(Path("models/scalar.xml"))
        self.assertEqual(result.inputs[0].rank, 0)
        self.assertEqual(result.inputs[0].shape, ())
        self.assertEqual(result.supported_schemas, ())

    def test_nonzero_inspector_exit_is_a_model_error(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = self.run_inspector(directory, "", exit_code=1)
            with self.assertRaisesRegex(NCS2ModelInspectionError, "exit status 1"):
                NCS2ModelInspector(executable).inspect(Path("models/bad.xml"))

    def test_native_inspector_reads_ir_without_opening_usb(self):
        inspector = ROOT / "build-arm64/ncs2_model_inspect"
        if not inspector.is_file():
            self.skipTest("Build native tools with ./ncs2 build-probe")
        xml = '''<net name="relu_1x4" version="10">
<layers>
<layer id="0" name="input" type="Parameter" version="opset1">
<data shape="1,4" element_type="f32"/>
<output><port id="0" precision="FP32"><dim>1</dim><dim>4</dim></port></output>
</layer>
<layer id="1" name="relu" type="ReLU" version="opset1">
<data alpha="0" beta="0"/>
<input><port id="0" precision="FP32"><dim>1</dim><dim>4</dim></port></input>
<output><port id="1" precision="FP32"><dim>1</dim><dim>4</dim></port></output>
</layer>
<layer id="2" name="result" type="Result" version="opset1">
<input><port id="0" precision="FP32"><dim>1</dim><dim>4</dim></port></input>
</layer>
</layers>
<edges>
<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
</edges>
</net>
'''
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "relu.xml"
            model.write_text(xml)
            output = NCS2ModelInspector(inspector).inspect(model)
            self.assertEqual(output.openvino_version, "2022.3.2")
            self.assertTrue(output.schema_1.compatible)
            self.assertTrue(output.schema_2.compatible)
            self.assertEqual(output.inputs[0].shape, (1, 4))
            self.assertFalse(output.hardware_accessed)
            log = Path(directory) / "notebook.md"
            result = subprocess.run([str(ROOT / "ncs2"), "inspect", "--model", str(model)],
                                   capture_output=True, text=True, timeout=10,
                                   env={**os.environ, "NCS2_LOG_FILE": str(log)})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)["hardware_accessed"])
            matrix = subprocess.run([str(ROOT / "ncs2"), "capabilities", "--json"],
                                    capture_output=True, text=True, timeout=10,
                                    env={**os.environ, "NCS2_LOG_FILE": str(log)})
            self.assertEqual(matrix.returncode, 0, matrix.stderr)
            self.assertFalse(json.loads(matrix.stdout)["protocol"]["cpu_fallback"])


class CapabilityMatrixTests(unittest.TestCase):
    def test_matrix_states_worker_limits_and_native_boundary(self):
        value = capabilities()
        self.assertEqual(value["host"]["architecture"], "arm64")
        self.assertEqual(value["models"]["dtype"], ["f32"])
        self.assertEqual(value["models"]["tensor_elements"]["aggregate_input_max"], 16384)
        self.assertFalse(value["models"]["dynamic_shapes"])
        self.assertFalse(value["protocol"]["cpu_fallback"])
        value["models"]["dtype"].append("i8")
        self.assertEqual(capabilities()["models"]["dtype"], ["f32"])


if __name__ == "__main__":
    unittest.main()

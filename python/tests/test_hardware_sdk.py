"""Opt-in hardware contract check for schema-2 SDK timing and reuse."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ncs2_async import NCS2Runtime


RELU_IR = '''<net name="relu_1x4" version="10">
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


@unittest.skipUnless(os.environ.get("NCS2_RUN_HARDWARE_TESTS") == "1",
                     "Set NCS2_RUN_HARDWARE_TESTS=1 to access the physical NCS2")
class HardwareSDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema2_inference_timer_worker_reuse_and_results(self):
        with tempfile.TemporaryDirectory(prefix="neuralstick-sdk-hardware-") as directory:
            model = Path(directory) / "relu.xml"
            model.write_text(RELU_IR)
            runtime = NCS2Runtime(
                worker=ROOT / "build-arm64/ncs2_worker",
                multi_worker=ROOT / "build-arm64/ncs2_multi_worker",
                plugins=ROOT / "vendor/openvino-2022.3/bin/arm64/Release/plugins.xml",
                max_retained_results=8,
                max_retained_elements=32,
            )
            async with runtime:
                await runtime.register_model("relu", model, schema=2)
                handles = [await runtime.submit("relu", [[-1, 0.5, -2, 3]]) for _ in range(5)]
                results = [await handle.result() for handle in handles]
                self.assertTrue(all(result.outputs[0].data == (0, 0.5, 0, 3) for result in results))
                self.assertTrue(all(result.inference_milliseconds is not None
                                    and result.inference_milliseconds >= 0 for result in results))
                state = await runtime.status()
                self.assertEqual(state.metrics.completed_tasks, 5)
                self.assertEqual(state.metrics.worker_start_count, 1)
                self.assertEqual(state.metrics.model_switch_count, 0)
                self.assertEqual(state.metrics.retained_result_count, 5)
                for handle in handles:
                    await handle.release()
                state = await runtime.status()
                self.assertEqual(state.metrics.retained_result_count, 0)


if __name__ == "__main__":
    unittest.main()

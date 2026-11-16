"""Native device ownership is enforced before touching MYRIAD or model files."""
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from ncs2_async import NCS2Runtime, NCS2SDKError, NCS2ErrorCode

ROOT = Path(__file__).resolve().parents[2]


class NativeLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_child_uses_device_lock_and_recovers_after_contention(self):
        worker = ROOT / 'build-arm64/ncs2_worker'
        if not worker.exists():
            self.skipTest('Build native workers with ./ncs2 build-probe')
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / 'device.lock'
            with lock.open('w') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                async with NCS2Runtime(worker=worker, multi_worker=worker,
                                       plugins=ROOT/'vendor/openvino-2022.3/bin/arm64/Release/plugins.xml',
                                       device_lock=lock) as runtime:
                    await runtime.register_model('missing', Path(directory)/'missing.xml', schema=1)
                    task = await runtime.submit('missing', [[1]])
                    with self.assertRaises(NCS2SDKError) as caught:
                        await task.result()
                    self.assertEqual(caught.exception.code, NCS2ErrorCode.DEVICE_BUSY)
                    fcntl.flock(owner, fcntl.LOCK_UN)
                    task = await runtime.submit('missing', [[1]])
                    with self.assertRaises(NCS2SDKError) as caught:
                        await task.result()
                    self.assertEqual(caught.exception.code, NCS2ErrorCode.MODEL_LOAD_FAILED)

    async def test_probe_and_both_workers_refuse_held_device_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory)/'device.lock'
            with lock.open('w') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for name in ('ncs2_worker', 'ncs2_multi_worker', 'ncs2_openvino_probe'):
                    binary = ROOT/'build-arm64'/name
                    if not binary.exists():
                        self.skipTest('Build native workers with ./ncs2 build-probe')
                    args = ['unused', 'list'] if name.endswith('probe') else ['unused', '--model', 'unused.xml']
                    proc = subprocess.run([str(binary), *args], env={**os.environ,'NCS2_DEVICE_LOCK':str(lock)},
                                          capture_output=True,text=True,timeout=5)
                    self.assertEqual(proc.returncode,3,proc.stderr)
                    self.assertIn('device_busy',proc.stderr)

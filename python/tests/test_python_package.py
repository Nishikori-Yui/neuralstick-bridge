"""The installable facade preserves the pre-package import surface."""
import importlib.util
from pathlib import Path
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PythonPackageTests(unittest.TestCase):
    def test_facade_exports_the_original_sdk_types(self):
        sys.path.insert(0, str(ROOT / "python"))
        self.addCleanup(sys.path.remove, str(ROOT / "python"))
        from neuralstick_bridge import NCS2Runtime, NCS2Worker, NCS2MultiWorker
        from ncs2_async import NCS2Runtime as Runtime
        from ncs2_client import NCS2Worker as Worker
        from ncs2_multi_client import NCS2MultiWorker as MultiWorker
        self.assertIs(NCS2Runtime, Runtime)
        self.assertIs(NCS2Worker, Worker)
        self.assertIs(NCS2MultiWorker, MultiWorker)

    def test_package_selects_only_sdk_modules(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(set(config["tool"]["setuptools"]["py-modules"]),
                         {"ncs2_client", "ncs2_multi_client", "ncs2_async"})
        self.assertNotIn("dependencies", config["project"])

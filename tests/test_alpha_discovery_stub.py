# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
import unittest
from pathlib import Path
import tempfile

STUB = "alpha_factory_v1/demos/omni_factory_demo/alpha_discovery_stub.py"


class TestAlphaDiscoveryStub(unittest.TestCase):
    def test_list_option(self) -> None:
        result = subprocess.run([sys.executable, STUB, "--list"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 5)

    def test_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "log.json"
            result = subprocess.run(
                [sys.executable, STUB, "-n", "2", "--seed", "1", "--ledger", str(ledger)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(ledger.exists())
            logged = json.loads(ledger.read_text())
            self.assertIsInstance(logged, list)
            self.assertEqual(len(logged), 2)
            self.assertEqual(len(logged), len({json.dumps(i, sort_keys=True) for i in logged}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

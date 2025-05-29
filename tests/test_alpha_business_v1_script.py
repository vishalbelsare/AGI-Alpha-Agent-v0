# SPDX-License-Identifier: Apache-2.0
import py_compile
import unittest
from pathlib import Path

class TestAlphaBusinessV1Script(unittest.TestCase):
    def test_script_compiles(self) -> None:
        path = Path('alpha_factory_v1/demos/alpha_agi_business_v1/alpha_agi_business_v1.py')
        py_compile.compile(path, doraise=True)

    def test_launcher_compiles(self) -> None:
        path = Path('alpha_factory_v1/demos/alpha_agi_business_v1/run_business_v1_local.py')
        py_compile.compile(path, doraise=True)

    def test_helper_compiles(self) -> None:
        """Ensure the one-click helper launcher compiles."""
        path = Path('alpha_factory_v1/demos/alpha_agi_business_v1/start_alpha_business.py')
        py_compile.compile(path, doraise=True)

if __name__ == '__main__':
    unittest.main()

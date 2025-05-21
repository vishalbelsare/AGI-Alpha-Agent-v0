import subprocess
import sys
import unittest


class TestAlphaAgiInsightMain(unittest.TestCase):
    """Verify the package entry point."""

    def test_main_offline(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alpha_factory_v1.demos.alpha_agi_insight_v0",
                "--offline",
                "--episodes",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Best sector", result.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

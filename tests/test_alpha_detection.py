# SPDX-License-Identifier: Apache-2.0
import unittest

from alpha_factory_v1.demos.era_of_experience import alpha_detection


class TestAlphaDetection(unittest.TestCase):
    def test_detect_yield_curve_alpha(self) -> None:
        msg = alpha_detection.detect_yield_curve_alpha()
        self.assertIsInstance(msg, str)
        self.assertTrue("Yield curve spread" in msg)

    def test_detect_supply_chain_alpha(self) -> None:
        msg = alpha_detection.detect_supply_chain_alpha()
        self.assertIsInstance(msg, str)
        # The message should either mention "USD" or indicate that "offline data" is missing.
        self.assertTrue(
            "USD" in msg or "offline data missing" in msg,
            "Expected 'USD' or 'offline data missing' in the message.",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`SafetyGuardianAgent` and ledger broadcasting."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import pytest

_STUB = "alpha_factory_v1.demos.alpha_agi_insight_v1.src.utils.a2a_pb2"
if _STUB not in sys.modules:
    stub = types.ModuleType("a2a_pb2")

    from dataclasses import dataclass

    @dataclass
    class Envelope:
        sender: str = ""
        recipient: str = ""
        payload: dict[str, object] | None = None
        ts: float = 0.0

    stub.Envelope = Envelope
    sys.modules[_STUB] = stub

from alpha_factory_v1.demos.alpha_agi_insight_v1.src.agents import safety_agent
from alpha_factory_v1.demos.alpha_agi_insight_v1.src.utils import logging as insight_logging
from alpha_factory_v1.demos.alpha_agi_insight_v1.src.utils import config, messaging


class DummyBus:
    def __init__(self, settings: config.Settings) -> None:
        self.settings = settings
        self.published: list[tuple[str, messaging.Envelope]] = []

    def publish(self, topic: str, env: messaging.Envelope) -> None:
        self.published.append((topic, env))

    def subscribe(self, topic: str, handler) -> None:  # pragma: no cover - dummy
        pass


class DummyLedger:
    def __init__(self) -> None:
        self.logged: list[messaging.Envelope] = []

    def log(self, env: messaging.Envelope) -> None:  # type: ignore[override]
        self.logged.append(env)

    def start_merkle_task(self, *a, **kw) -> None:  # pragma: no cover - dummy
        pass

    async def stop_merkle_task(self) -> None:  # pragma: no cover - interface
        pass

    def close(self) -> None:  # pragma: no cover - dummy
        pass


class TestSafetyGuardian(unittest.TestCase):
    """Validate blocking behaviour of :class:`SafetyGuardianAgent`."""

    def setUp(self) -> None:
        cfg = config.Settings(bus_port=0)
        self.bus = DummyBus(cfg)
        self.ledger = DummyLedger()
        self.agent = safety_agent.SafetyGuardianAgent(self.bus, self.ledger)

    def test_blocks_malicious_code(self) -> None:
        env = messaging.Envelope("codegen", "safety", {"code": "import os\nos.system('rm -rf /')"}, 0.0)
        asyncio.run(self.agent.handle(env))
        self.assertEqual(self.bus.published[-1][1].payload["status"], "blocked")

    def test_allows_safe_code(self) -> None:
        env = messaging.Envelope("codegen", "safety", {"code": "print('hi')"}, 0.0)
        asyncio.run(self.agent.handle(env))
        self.assertEqual(self.bus.published[-1][1].payload["status"], "ok")


@pytest.mark.skipif(os.getenv("PYTEST_NET_OFF") == "1", reason="network disabled")
class TestLedgerBroadcast(unittest.TestCase):
    """Verify Merkle root broadcasting when network is available."""

    def _ledger(self) -> insight_logging.Ledger:
        tmp = tempfile.TemporaryDirectory()
        led = insight_logging.Ledger(os.path.join(tmp.name, "l.db"), rpc_url="http://rpc.test", broadcast=True)
        self.addCleanup(tmp.cleanup)
        return led

    def _dummy_classes(self):
        captured = {}

        class DummyClient:
            def __init__(self, url: str) -> None:
                captured["url"] = url

            async def send_transaction(self, tx: object, *args: object) -> None:
                captured["data"] = tx.instructions[0].data.decode()

            async def close(self) -> None:  # pragma: no cover - dummy
                pass

        class DummyTx:
            def __init__(self) -> None:
                self.instructions = []

            def add(self, instr: object) -> "DummyTx":
                self.instructions.append(instr)
                return self

        class DummyInstr:
            def __init__(self, program_id: object, data: bytes, keys: list[object]):
                self.data = data

        class DummyPk:
            def __init__(self, val: str) -> None:  # pragma: no cover - dummy
                pass

        return captured, DummyClient, DummyTx, DummyInstr, DummyPk

    def test_broadcast_merkle_root(self) -> None:
        led = self._ledger()
        env = messaging.Envelope("a", "b", {"v": 1}, 0.0)
        led.log(env)
        root = led.compute_merkle_root()
        captured, DummyClient, DummyTx, DummyInstr, DummyPk = self._dummy_classes()
        with (
            mock.patch.object(insight_logging, "AsyncClient", DummyClient, create=True),
            mock.patch.object(insight_logging, "Transaction", DummyTx, create=True),
            mock.patch.object(insight_logging, "TransactionInstruction", DummyInstr, create=True),
            mock.patch.object(insight_logging, "PublicKey", DummyPk, create=True),
        ):
            asyncio.run(led.broadcast_merkle_root())
        self.assertEqual(captured["url"], "http://rpc.test")
        self.assertEqual(captured["data"], root)

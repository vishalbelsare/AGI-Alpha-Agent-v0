# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
from unittest import mock

import pytest

from alpha_factory_v1.common.utils import config, messaging


def test_bus_logs_start_stop(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    cfg = config.Settings(bus_port=1234, broker_url="kafka:9092")
    with mock.patch.object(messaging, "AIOKafkaProducer", None), mock.patch.object(messaging, "grpc", None):

        async def run() -> None:
            async with messaging.A2ABus(cfg):
                pass

        asyncio.run(run())
    messages = [r.message for r in caplog.records]
    assert any("A2ABus.start()" in m and "1234" in m and "kafka:9092" in m for m in messages)
    assert any("A2ABus.stop()" in m for m in messages)

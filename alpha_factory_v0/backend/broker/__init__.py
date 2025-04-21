# backend/broker/__init__.py
import os

_BROKER = os.getenv("ALPHA_BROKER", "sim").lower()

if _BROKER == "alpaca":
    from .broker_alpaca import AlpacaBroker as Broker  # type: ignore
elif _BROKER == "ibkr":
    from .broker_ibkr import InteractiveBrokersBroker as Broker  # type: ignore
else:
    from .broker_sim import SimulatedBroker as Broker  # type: ignore

import os, logging, requests, random, time
from datetime import datetime

log = logging.getLogger("Broker")

ALPACA_BASE = "https://paper-api.alpaca.markets/v2"

class Order:
    def __init__(self, symbol, qty, side, price):
        self.symbol, self.qty, self.side, self.price = symbol, qty, side, price
        self.ts = datetime.utcnow().isoformat()
        self.status = "filled"

def _alpaca_order(symbol, qty, side, key, secret):
    hdrs = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    data = {"symbol": symbol, "qty": qty, "side": side.lower(), "type": "market", "time_in_force": "gtc"}
    r = requests.post(f"{ALPACA_BASE}/orders", json=data, headers=hdrs, timeout=4)
    r.raise_for_status()
    return r.json()

# ── public API ───────────────────────────────────────────────────────────
def place_order(symbol: str, qty: int, side: str, price: float):
    """
    Execute an order.  Priority:
      1. Alpaca paper‑trading (ALPACA_KEY_ID + ALPACA_SECRET_KEY)
      2. Offline simulator – instant fill, log only
    Returns an Order‑like dict.
    """
    key = os.getenv("ALPACA_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY")
    try:
        if key and secret:
            return _alpaca_order(symbol, qty, side, key, secret)
    except Exception as err:
        log.warning("Live broker failed (%s); falling back to simulator.", err)

    # stub fill
    time.sleep(0.1)
    return Order(symbol, qty, side, price).__dict__


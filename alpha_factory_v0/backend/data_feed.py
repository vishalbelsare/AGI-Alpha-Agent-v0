import os, time, random, requests, logging
from datetime import datetime

log = logging.getLogger("DataFeed")

# ── helpers ──────────────────────────────────────────────────────────────
def _polygon_last_price(symbol: str, key: str) -> float:
    url = f"https://api.polygon.io/v2/last/trade/{symbol}?apiKey={key}"
    r = requests.get(url, timeout=4).json()
    return r["results"]["p"]  # price

def _yahoo_last_price(symbol: str) -> float:
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=price"
    p = requests.get(url, timeout=4).json()["quoteSummary"]["result"][0]["price"]
    return p["regularMarketPrice"]["raw"]

# ── public API ───────────────────────────────────────────────────────────
def last_price(symbol: str) -> float:
    """
    Return latest trade/quote for *symbol*.
    Auto‑selects in this priority:
      1. Polygon (POLYGON_API_KEY)
      2. Yahoo Finance (no key)
      3. Offline random‑walk stub
    """
    poly_key = os.getenv("POLYGON_API_KEY")
    try:
        if poly_key:
            return _polygon_last_price(symbol, poly_key)
        return _yahoo_last_price(symbol)
    except Exception as err:
        log.warning("Live feed failed (%s); using stub.", err)
        # random‑walk around 100
        return 100 + random.gauss(0, 0.5)


package codegen

# Deny outbound finance APIs
banned_hosts = {
    "api.alpaca.markets",
    "api.binance.com",
    "api.polygon.io",
    "api.kraken.com",
    "api.coinbase.com",
    "api.gemini.com"
}

deny[msg] {
    some host
    host := banned_hosts[_]
    input.url == host
    msg := "finance API blocked"
}

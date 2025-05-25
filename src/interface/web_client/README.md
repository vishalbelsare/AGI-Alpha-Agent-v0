# Web Client

Minimal React client visualizing the simulation API. Start `api_server.py` with:

```bash
uvicorn src.interface.api_server:app --reload
```

Serve this directory (for example with `python -m http.server`) and open
`index.html` in your browser. The app streams logs over WebSocket and renders
capability and Pareto charts using Plotly.

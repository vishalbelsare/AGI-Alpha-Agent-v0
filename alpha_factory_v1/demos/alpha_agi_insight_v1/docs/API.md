# API Overview

This demo exposes a minimal REST and WebSocket interface implemented in `src/interface/api_server.py`. All requests must include `Authorization: Bearer $API_TOKEN`.

| Endpoint | Query/Path Params | Payload | Example Response |
|---------|------------------|---------|-----------------|
| **POST `/simulate`** | – | `{ "horizon": 5, "pop_size": 6, "generations": 3 }` | `{ "id": "d4e5f6a7" }` |
| **GET `/results/{sim_id}`** | `sim_id` (path) | – | `{ "id": "d4e5f6a7", "forecast": [{"year": 1, "capability": 0.1}] }` |
| **GET `/runs`** | – | – | `{ "ids": ["d4e5f6a7"] }` |
| **WS `/ws/progress`** | – | N/A | `{"id": "d4e5f6a7", "year": 1, "capability": 0.1}` |


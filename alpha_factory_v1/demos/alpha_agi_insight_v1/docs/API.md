# API Overview

This demo exposes a minimal REST and WebSocket interface implemented in `src/interface/api_server.py`. All requests must include `Authorization: Bearer $API_TOKEN`.

| Endpoint | Query/Path Params | Payload | Example Response |
|---------|------------------|---------|-----------------|
| **POST `/simulate`** | – | `{ "horizon": 5, "pop_size": 6, "generations": 3 }` | `{ "id": "d4e5f6a7" }` |
| **GET `/results/{sim_id}`** | `sim_id` (path) | – | `{ "id": "d4e5f6a7", "forecast": [{"year": 1, "capability": 0.1}], "population": [] }` |
| **GET `/population/{sim_id}`** | `sim_id` (path) | – | `{ "id": "d4e5f6a7", "population": [] }` |
| **GET `/runs`** | – | – | `{ "ids": ["d4e5f6a7"] }` |
| **WS `/ws/progress`** | – | N/A | `{"id": "d4e5f6a7", "year": 1, "capability": 0.1}` |

### `GET /population/{sim_id}`

Retrieve only the final population for a completed run.

```bash
curl -H "Authorization: Bearer $API_TOKEN" \
  http://localhost:8000/population/<sim_id>
```

Example response:

```json
{
  "id": "<sim_id>",
  "population": [
    {"effectiveness": 0.5, "risk": 0.2, "complexity": 0.3, "rank": 0}
  ]
}
```


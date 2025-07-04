"""FastAPI server exposing simulation endpoints."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import secrets
from typing import Any, List, TYPE_CHECKING, cast, Set


if TYPE_CHECKING:  # pragma: no cover - typing only
    from typing import Protocol

    class ForecastTrajectoryPoint(Protocol):
        year: int
        capability: float
        sectors: List[Any]

    ForecastModule = Any
    SectorModule = Any
    MatsModule = Any
else:
    ForecastModule = Any
    SectorModule = Any
    MatsModule = Any
    ForecastTrajectoryPoint = Any  # type: ignore[assignment]

forecast = importlib.import_module("alpha_factory_v1.demos.alpha_agi_insight_v1.src.simulation.forecast")
sector = importlib.import_module("alpha_factory_v1.demos.alpha_agi_insight_v1.src.simulation.sector")

_IMPORT_ERROR: Exception | None
try:
    from fastapi import FastAPI, HTTPException, WebSocket
    from pydantic import BaseModel
    import uvicorn
except Exception as exc:  # pragma: no cover - optional
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    BaseModel = object  # type: ignore
    WebSocket = Any  # type: ignore
    uvicorn = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

app: FastAPI | None = FastAPI(title="AGI Simulation API") if FastAPI is not None else None

if app is not None:
    app_f: FastAPI = app
    _orch: Any | None = None

    @app.on_event("startup")
    async def _start() -> None:
        global _orch
        orch_mod = importlib.import_module("alpha_factory_v1.demos.alpha_agi_insight_v1.src.orchestrator")
        _orch = orch_mod.Orchestrator()
        app_f.state.orch_task = asyncio.create_task(_orch.run_forever())

    @app.on_event("shutdown")
    async def _stop() -> None:
        global _orch
        task = getattr(app_f.state, "orch_task", None)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        _orch = None


_simulations: dict[str, list[ForecastTrajectoryPoint]] = {}
_progress_ws: Set[Any] = set()


class SimRequest(BaseModel):
    """Payload for the ``/simulate`` endpoint."""

    horizon: int = 5
    pop_size: int = 6
    generations: int = 3


async def _background_run(sim_id: str, cfg: SimRequest) -> None:
    """Execute one simulation in the background.

    Args:
        sim_id: Unique identifier for the run.
        cfg: Parameters controlling the forecast generation.

    Returns:
        None
    """

    secs = [sector.Sector(f"s{i:02d}") for i in range(cfg.pop_size)]
    results: list[ForecastTrajectoryPoint] = []
    for year in range(1, cfg.horizon + 1):
        t = year / cfg.horizon
        cap = forecast.capability_growth(t)
        for sec in secs:
            if not sec.disrupted:
                sec.energy *= 1.0 + sec.growth
                if forecast.thermodynamic_trigger(sec, cap):
                    sec.disrupted = True
                    sec.energy += forecast._innovation_gain(cfg.pop_size, cfg.generations)
        snapshot = [sector.Sector(s.name, s.energy, s.entropy, s.growth, s.disrupted) for s in secs]
        point = forecast.TrajectoryPoint(year, cap, snapshot)
        results.append(point)
        for ws in list(_progress_ws):
            try:
                await ws.send_json({"id": sim_id, "year": year, "capability": cap})
            except Exception:
                _progress_ws.discard(ws)
        await asyncio.sleep(0)
    _simulations[sim_id] = results


if app is not None:

    @app.post("/simulate")
    async def simulate(req: SimRequest) -> dict[str, str]:
        """Start a simulation and return its identifier.

        Args:
            req: Simulation request parameters.

        Returns:
            A mapping containing the ``id`` of the background run.
        """
        sim_id = secrets.token_hex(8)
        asyncio.create_task(_background_run(sim_id, req))
        return {"id": sim_id}

    @app.get("/results/{sim_id}")
    async def get_results(sim_id: str) -> dict[str, Any]:
        """Return final forecast data for ``sim_id`` if available."""
        traj = _simulations.get(sim_id)
        if traj is None:
            raise HTTPException(status_code=404)
        data = [{"year": t.year, "capability": t.capability} for t in traj]
        return {"id": sim_id, "forecast": data}

    @app.websocket("/ws/progress")
    async def ws_progress(websocket: WebSocket) -> None:
        """Stream year-by-year progress updates to the client."""
        await websocket.accept()
        _progress_ws.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except Exception:
            pass
        finally:
            _progress_ws.discard(websocket)


def main(argv: List[str] | None = None) -> None:
    """CLI entry to launch the API server.

    Args:
        argv: Optional list of command line arguments.

    Returns:
        None
    """
    if FastAPI is None:
        raise SystemExit("FastAPI is required to run the API server") from _IMPORT_ERROR

    parser = argparse.ArgumentParser(description="Run the AGI simulation API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args(argv)
    uvicorn.run(cast(Any, app), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

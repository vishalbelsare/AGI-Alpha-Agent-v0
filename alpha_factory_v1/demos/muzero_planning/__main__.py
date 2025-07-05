# SPDX-License-Identifier: Apache-2.0
"""Command line entry point for the MuZero planning demo."""

from __future__ import annotations

import argparse
import os
from ...utils.disclaimer import print_disclaimer


def main(argv: list[str] | None = None) -> None:
    """Launch the MuZero dashboard with optional CLI overrides."""

    print_disclaimer()

    parser = argparse.ArgumentParser(description="Run MuZero planning demo")
    parser.add_argument(
        "--env",
        default=os.getenv("MUZERO_ENV_ID", "CartPole-v1"),
        help="Gymnasium environment ID",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.getenv("MUZERO_EPISODES", 3)),
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("HOST_PORT", 7861)),
        help="Dashboard port",
    )
    args = parser.parse_args(argv)

    os.environ["MUZERO_ENV_ID"] = args.env
    os.environ["MUZERO_EPISODES"] = str(args.episodes)
    os.environ["HOST_PORT"] = str(args.port)

    try:
        from .agent_muzero_entrypoint import launch_dashboard
    except Exception as exc:  # pragma: no cover - optional deps missing
        print(f"MuZero demo requires optional packages: {exc}")
        return

    launch_dashboard()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

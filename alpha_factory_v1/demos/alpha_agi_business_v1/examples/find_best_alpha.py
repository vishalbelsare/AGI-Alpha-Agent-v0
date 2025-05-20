import json
from pathlib import Path


def main() -> None:
    """Print the highest scoring alpha opportunity."""
    path = Path(__file__).with_name("alpha_opportunities.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    best = max(data, key=lambda x: x.get("score", 0))
    print("Best alpha opportunity:")
    print(f"  description: {best['alpha']}")
    print(f"  score: {best['score']}")


if __name__ == "__main__":
    main()

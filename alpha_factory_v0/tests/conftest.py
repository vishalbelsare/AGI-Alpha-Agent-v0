# Ensure the project root is importable when pytest cwd == tests/
import sys, pathlib, os

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # so relative paths (e.g. .env) resolve the same way


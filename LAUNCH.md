# Release Checklist

1. Run `pre-commit run --all-files`.
2. Run `python scripts/check_python_deps.py`.
3. Run `python check_env.py --auto-install`.
4. Execute `pytest -q` and ensure all tests pass.
5. Build the web client with `make build_web`.
6. Update `docs/CHANGELOG.md` with the new version.
7. Commit changes and tag the release: `git tag -s vX.Y.Z -m "vX.Y.Z"`.
8. Push commits and tags to GitHub.
9. The `CI` workflow builds the image and uploads release artifacts.

## Tweet Copy

> 🚀 New Alpha-Factory release! Offline dashboard, responsive UI and automated visual tests powered by Percy. Check it out: https://github.com/AGI-Factory/AGI-Alpha-Agent-v0

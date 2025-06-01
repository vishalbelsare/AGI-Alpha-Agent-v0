# Release Checklist

1. Run `pre-commit run --all-files`.
2. Run `python check_env.py --auto-install`.
3. Execute `pytest -q` and ensure all tests pass.
4. Build the web client with `make build_web`.
5. Update `docs/CHANGELOG.md` with the new version.
6. Commit changes and tag the release: `git tag -s vX.Y.Z -m "vX.Y.Z"`.
7. Push commits and tags to GitHub.
8. The `CI` workflow builds the image and uploads release artifacts.

## Tweet Copy

> 🚀 New Alpha-Factory release! Offline dashboard, responsive UI and automated visual tests powered by Percy. Check it out: https://github.com/AGI-Factory/AGI-Alpha-Agent-v0

[See docs/DISCLAIMER_SNIPPET.md](docs/DISCLAIMER_SNIPPET.md)

# Release Checklist

1. Run `pre-commit run --all-files`.
2. Run `python scripts/check_python_deps.py`.
3. Run `python check_env.py --auto-install`.
4. Execute `pytest -q` and ensure all tests pass.
5. Build the web client with `make build_web`.
6. Update `docs/CHANGELOG.md` with the new version.
7. Commit changes and tag the release using `./scripts/create_release_tag.sh <commit>`.
   The helper creates the annotated tag `v0.1.0-alpha` (defaults to `HEAD` when
   no commit is provided). Ensure `pre-commit run --all-files`,
   `python check_env.py --auto-install` and `pytest -q` all succeed before
   creating the tag. When offline, pass `--wheelhouse <dir>` to `check_env.py`
   and run the tests from that wheelhouse.
8. Push commits and tags to GitHub.
9. The `CI` workflow builds the image and uploads release artifacts.

## Tweet Copy

> 🚀 New Alpha-Factory release! Offline dashboard, responsive UI and automated visual tests powered by Percy. Check it out: https://github.com/AGI-Factory/AGI-Alpha-Agent-v0

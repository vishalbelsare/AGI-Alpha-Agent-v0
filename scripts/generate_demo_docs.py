#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate documentation pages for each demo in docs/demos.

This utility scans the ``alpha_factory_v1/demos`` directory for subpackages
containing ``README.md`` files and creates matching Markdown pages under
``docs/demos``. Each generated page embeds the project disclaimer,
links back to the original README and optionally displays a preview
image found under ``docs/<demo>/assets/preview.*``.

Run this script whenever a new demo is added or READMEs change so the
GitHub Pages gallery stays up to date.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMOS_DIR = REPO_ROOT / "alpha_factory_v1" / "demos"
DOCS_DIR = REPO_ROOT / "docs" / "demos"
DEFAULT_PREVIEW = "../alpha_agi_insight_v1/favicon.svg"
DISCLAIMER_LINK = "[See docs/DISCLAIMER_SNIPPET.md](../DISCLAIMER_SNIPPET.md)"

TITLE_RE = re.compile(r"^#(?!#)\s*(.+)")


def extract_title(readme: Path) -> str:
    """Return a reasonable title for the given README."""
    lines = readme.read_text(encoding="utf-8").splitlines()
    # Search the first 50 lines for a level-one heading
    for line in lines[:50]:
        m = TITLE_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    # Fallback to folder name if no heading found early in the file
    return readme.parent.name.replace("_", " ").title()


def build_page(demo: Path) -> str:
    """Return Markdown text for the given demo subdirectory."""
    title = extract_title(demo / "README.md")
    assets_dir = REPO_ROOT / "docs" / demo.name / "assets"
    preview = None
    if assets_dir.is_dir():
        for ext in ("gif", "png", "jpg", "jpeg", "svg"):
            candidate = assets_dir / f"preview.{ext}"
            if candidate.exists():
                preview = f"../{demo.name}/assets/{candidate.name}"
                break
    if not preview:
        preview = DEFAULT_PREVIEW

    launch_link = None
    demo_index = REPO_ROOT / "docs" / demo.name / "index.html"
    if demo_index.is_file():
        launch_link = f"[Launch Demo](../{demo.name}/){{.md-button}}"

    readme_path = demo / "README.md"
    readme_lines = readme_path.read_text(encoding="utf-8").splitlines()
    if readme_lines and readme_lines[0].startswith("#"):
        readme_lines = readme_lines[1:]

    cleaned: list[str] = []
    skip_section = False
    for line in readme_lines:
        stripped = line.strip()
        if skip_section:
            if stripped.startswith("#") or stripped.startswith("---") or not stripped:
                skip_section = False
            continue
        if "DISCLAIMER_SNIPPET.md" in stripped:
            continue
        if (
            "conceptual research prototype" in stripped
            or "financial advice" in stripped
            or "research and educational purposes" in stripped
            or "trading decisions" in stripped
            or "no liability" in stripped
        ):
            continue
        if stripped.lower().startswith("##") and "disclaimer" in stripped.lower():
            skip_section = True
            continue
        cleaned.append(line)

    readme_text = "\n".join(cleaned).lstrip("\n")

    content = [
        DISCLAIMER_LINK,
        "",
        f"# {title}",
        "",
        f"![preview]({preview}){{.demo-preview}}",
    ]
    if launch_link:
        content.extend(["", launch_link, ""])
    content.extend([
        readme_text,
        "",
        f"[View README](../../alpha_factory_v1/demos/{demo.name}/README.md)",
        "",
    ])

    return "\n".join(content)


def generate_docs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for entry in sorted(DEMOS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("__", ".")):
            continue
        readme = entry / "README.md"
        if not readme.is_file():
            continue
        page_content = build_page(entry)
        output = DOCS_DIR / f"{entry.name}.md"
        output.write_text(page_content, encoding="utf-8")
        print(f"Generated {output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    generate_docs()


# Meta‑Agentic α‑AGI UI Assets

> **Directory:** `alpha_factory_v1/demos/meta_agentic_agi/ui/assets/`  
> **Version:** `v0.5.0  (2025‑05‑05)`  
> **Status:** Stable · Production‑ready

This folder contains **all static front‑end assets** used by the Meta‑Agentic α‑AGI demo—icons, illustrations, logos, fonts, audio cues, colour tokens, and micro‑interaction sprites.  
Assets are **pipeline‑optimised, accessible, theme‑aware, and version‑controlled**, so they can be consumed interchangeably whether you launch the demo with closed‑weight LLM‑APIs (OpenAI GPT‑4o, Anthropic Opus) or fully‑local open‑weights (e.g. Llama‑3 70B via vLLM).

---

## 1 · Quick Start — Using an Asset

```python
from alpha_factory_v1.meta_agentic_agi.ui.helpers import asset_url

# reference the dark‑mode SVG for the “research” icon
st.image(asset_url("icons/research.svg"))
```

| Helper | Description |
|--------|-------------|
| `asset_url(name: str)` | Returns an absolute URL or local file path (depending on deployment mode). |
| `theme_token(key: str)` | Access colour / spacing design‑tokens (auto‑switches light ⚪ / dark ⚫). |

---

## 2 · Folder Layout

```
assets/
├─ icons/              # ∼32×32 SVGs for nav & toolbars
│   ├─ research.svg
│   ├─ strategist.svg
│   └─ …
├─ illu/               # High‑resolution hero illustrations (PNG+LQIP)
├─ sprites/            # GIF / APNG micro animations
├─ fonts/
│   ├─ InterVariable.woff2
│   └─ JetBrainsMono.woff2
├─ tokens.json         # Design‑token source‑of‑truth
└─ LICENSES/
    └─ …
```

*All SVGs are pre‑optimised with **SVGO v3** using the default “preset‑default” + `--enable removeViewBox`.  
Bitmap items are run through **squoosh-cli** (`oxipng + mozjpeg`) with **LQIP** companions for instant progressive loading.*

---

## 3 · Design Tokens

`tokens.json` exposes a **single‑source‑of‑truth** for colour, typography, radius, elevation and motion primitives.

| Token | Light | Dark | Notes |
|-------|-------|------|-------|
| `--alpha‑violet‑700` | `#4A2AFF` | `#7F7BFF` | Brand accent, passes WCAG 2.2 AA on both themes |
| `--surface‑1` | `#FFFFFF` | `#10111A` | Card backgrounds |
| `--radius‑md` | `6px` | – | Border‑radius baseline |

**Never hard‑code colour hexes**—always reference the semantic token.

---

## 4 · Adding / Updating Assets

1. **Drop files** into the relevant sub‑folder.  
2. Run the asset pipeline:

```bash
cd alpha_factory_v1/demos/meta_agentic_agi/ui
python -m tools.build_assets
```

> This lints SVGs, compresses bitmaps, updates `tokens.json`, bumps semantic version, and writes a provenance entry to the **Lineage DB** for end‑to‑end auditability.

3. Commit **both** the optimised asset *and* the auto‑generated `manifest‑*.json`.

---

## 5 · Accessibility & Internationalisation

* Every icon has a **descriptive `<title>`** node for screen‑readers.  
* All palette combinations meet **WCAG 2.2 AA** contrast ratios.  
* Font‑fallback stack includes full **CJK + RTL** coverage.  
* Sprites avoid red/green motion for colour‑blind comfort.

---

## 6 · Theming & Runtime Customisation

The Streamlit UI hot‑loads `tokens.json` at startup; switching themes triggers a CSS variable swap with zero reload.  
If you serve the demo from a notebook or FastAPI, call:

```python
from alpha_factory_v1.meta_agentic_agi.ui.theming import set_theme
set_theme("solarized")   # or "system", "dark", "light"
```

---

## 7 · Licensing

| Asset Family | License | Attribution |
|--------------|---------|-------------|
| Icons (`/icons`) | MIT | © Montreal.AI Design |
| Illustrations (`/illu`) | CC‑BY‑4.0 | Link back to repo |
| Fonts (`/fonts`) | SIL OFL 1.1 | Original authors |

See individual files in `LICENSES/`.

---

## 8 · Security & Compliance

* Hashes recorded in `manifest‑$VERSION.json` are checked at runtime to block tampering.  
* All network‑fetched assets are **sub‑resource‑integrity** (SRI) stamped when deployed over CDN.  
* Provenance entries stored in the **Lineage DB** fulfil ISO‑42001 audit requirements.

---

## 9 · Contributing

1. Fork the repo & create a feature branch: `git checkout -b feat/new‑asset‑pack`.
2. Add assets + run the build pipeline.
3. Open a PR. CI will fail if:
   * Lint / optimisation budget exceeded.
   * Missing SRI digest.
   * Contrast/bounds check fails.

Need help? Ping `#alpha‑ui` on the community Discord.

---

## 10 · Changelog (excerpt)

| Date | Version | Notes |
|------|---------|-------|
| 2025‑05‑05 | **0.5.0** | Initial public release. |
| 2025‑05‑01 | 0.4.2 | Added “antifragile” sprite set & Solarized theme. |
| 2025‑04‑20 | 0.4.0 | Design‑token refactor; WCAG 2.2 compliance pass. |

---

### One‑Line Philosophy

> *“Clarity, performance, auditability—without compromise.”*

Welcome to the **Era of Experience**—enjoy building with Meta‑Agentic α‑AGI! 👁️✨

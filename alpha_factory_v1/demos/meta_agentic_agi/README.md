alpha_factory_v1/
└── demos/
    └── meta_agentic_agi/
        ├── demo.py              # entry-point (196 LOC)
        ├── objectives.py        # vector fitness helpers
        ├── provider.py          # OpenAI / Anthropic / open-weights shim
        ├── ui/
        │   ├── app.py           # FastAPI server (read-only)
        │   └── static/…         # Cytoscape, Tailwind, logos
        └── README.md

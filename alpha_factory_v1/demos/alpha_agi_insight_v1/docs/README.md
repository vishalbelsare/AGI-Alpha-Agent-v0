# α‑AGI Insight Docs

Documentation for the α‑AGI Insight demo.

Whenever `src/utils/a2a.proto` changes, regenerate the protocol stubs:

```bash
./tools/gen_proto_stubs.sh
```

This updates `src/utils/a2a_pb2.py` and `tools/go_a2a_client/a2a.pb.go`.

## Quickstart

From the `alpha_factory_v1/demos/alpha_agi_insight_v1` directory run the
environment check before launching the demo:

```bash
python ../../../check_env.py --auto-install
```

Start the Insight demo with the repository root
[quickstart.sh](../../../quickstart.sh):

```bash
../../../quickstart.sh
```

An interactive walkthrough is available in the Colab notebook
[colab_alpha_agi_insight_v1.ipynb](../colab_alpha_agi_insight_v1.ipynb).


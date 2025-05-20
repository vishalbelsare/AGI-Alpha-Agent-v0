# 🏭 Production Deployment Guide — Alpha‑AGI Business v1

This guide summarises the minimal steps required to run the **Alpha‑AGI Business v1** demo in a production‑like environment. The service works fully offline but upgrades automatically when `OPENAI_API_KEY` is present.

1. **Prepare the configuration**
   - Copy `config.env.sample` to `config.env` and edit as needed.
   - Set `OPENAI_API_KEY` to enable cloud models or leave empty to use the bundled local fallbacks.
   - Optionally enable the Google ADK gateway by setting `ALPHA_FACTORY_ENABLE_ADK=true`.
   - For API protection set either `AUTH_BEARER_TOKEN` or `JWT_PUBLIC_KEY`/`JWT_ISSUER`.
   - Validate that all Python packages are available. From the project root run:
     ```bash
     AUTO_INSTALL_MISSING=1 python check_env.py
     ```
     Provide `WHEELHOUSE=/path/to/wheels` for air‑gapped setups.

2. **Launch the service**
   - **Docker** (recommended for consistent environments):
     ```bash
     ./run_business_v1_demo.sh --pull        # add --gpu for NVIDIA runtime
     ```
   - **Native Python**:
     ```bash
     pip install -r ../../requirements.txt
     python run_business_v1_local.py --bridge
     ```

3. **Run in Colab**
   - Open [`colab_alpha_agi_business_v1_demo.ipynb`](colab_alpha_agi_business_v1_demo.ipynb).
   - Run the setup cell; dependencies are installed automatically.
   - The notebook exposes a Gradio dashboard and OpenAI Agents SDK bridge.

4. **Access the interface**
   - REST/Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Gradio dashboard: [http://localhost:7860](http://localhost:7860)
   - Prometheus metrics: [http://localhost:8000/metrics](http://localhost:8000/metrics)

5. **Shutting down**
   - Docker: `./run_business_v1_demo.sh --stop`
   - Native Python: press `Ctrl+C`; the orchestrator shuts down gracefully.

For advanced options see [`README.md`](README.md) in the same directory.

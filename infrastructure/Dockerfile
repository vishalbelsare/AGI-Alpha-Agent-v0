FROM python:3.11-slim
WORKDIR /app
COPY .. /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "-m", "alpha_factory_v1.demos.alpha_agi_insight_v0.official_demo_final"]

# Demo container for α‑AGI Insight
FROM python:3.11-slim

# Install build tools for optional native extensions
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies for the demo
COPY alpha_factory_v1/demos/alpha_agi_insight_v1/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy the project source
COPY . /app
COPY src/interface/web_client/dist /app/src/interface/web_client/dist

# Add non-root user and entrypoint
RUN adduser --disabled-password --gecos '' afuser && chown -R afuser /app
COPY infrastructure/docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
USER afuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8501 6006
ENTRYPOINT ["entrypoint.sh"]
CMD ["web"]

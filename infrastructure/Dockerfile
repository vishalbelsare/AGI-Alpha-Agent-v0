FROM python:3.11-slim

# install system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# copy project source
COPY . /app

# add non-root user
RUN adduser --disabled-password --gecos '' afuser && chown -R afuser /app
COPY infrastructure/docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
USER afuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8000 3000 6006
ENTRYPOINT ["entrypoint.sh"]
CMD ["cli"]

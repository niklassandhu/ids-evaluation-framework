FROM python:3.13-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# System library required by nfstream (libpcap) for pcap ingestion / live capture
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpcap0.8 \
    && rm -rf /var/lib/apt/lists/*

ARG EXTRAS=""

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY LICENSE README.md ./
COPY examples/run_config ./run_config
RUN uv sync --no-dev ${EXTRAS:+--extra ${EXTRAS}}

# Create empty directories (will be overwritten by volume mounts)
RUN mkdir -p plugin_ids plugin_static_metric plugin_runtime_metric plugin_adversarial logs \
    out/reports out/processed_datasets out/saved_models

ENV PYTHONPATH=/app/src
ENV PATH=/app/.venv/bin:$PATH

ENTRYPOINT ["ids-eval"]

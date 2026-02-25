FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

# Install minimal build deps and libraries often required by JAX
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata first to leverage Docker layer caching
COPY pyproject.toml README.md /workspace/
COPY poisson_topicmodels /workspace/poisson_topicmodels

# Install the package directly from pyproject.toml
RUN pip install --upgrade pip setuptools wheel
RUN pip install .

# Ensure JupyterLab is available for interactive use
RUN pip install jupyterlab

# Copy the full repository for notebooks/examples/data usage
COPY . /workspace

# Expose notebook port
EXPOSE 8888

# Start JupyterLab. By default token/password disabled for local dev (set via env var if needed)
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]

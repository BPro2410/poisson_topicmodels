FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

# Install minimal build deps and libraries often required by JAX
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel
# Install project's requirements (will fail if jax/jaxlib versions in requirements need platform-specific wheels)
RUN pip install -r /workspace/requirements.txt

# Ensure JupyterLab is available for interactive use
RUN pip install jupyterlab

# Copy the repository into the container (includes run_topicmodels.py, data/, packages/, ...)
COPY . /workspace

# Expose notebook port
EXPOSE 8888

# Start JupyterLab. By default token/password disabled for local dev (set via env var if needed)
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
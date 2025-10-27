# topicmodels_package

``topicmodels`` is a Python package for probabilistic topic modeling using
Bayesian inference built on `JAX <https://github.com/google/jax>`_ and `NumPyro <https://github.com/pyro-ppl/numpyro>`_.

It provides implementations of several advanced topic models:

- **Poisson Factorization (PF)** – unsupervised baseline topic model.
- **Seeded Poisson Factorization (SPF)** – guided topic modeling with keyword priors.
- **Covariate Poisson Factorization (CPF)** – models topics influenced by external covariates.
- **Covariate Seeded Poisson Factorization (CSPF)** – combines seeded guidance with covariate effects.
- **Text-Based Ideal Points (TBIP)** – estimates ideal points of authors from text.
- **Time-Varying Text-Based Ideal Points (TVTBIP)** – captures temporal dynamics in authors' ideal points.
- **Structual Text-Based Scaling (STBS)** – models text data with structural information.
- **Embedded Topic Models (ETM)** – integrates word embeddings into topic modeling.
- ... and more models to come! 


The package emphasizes **scalability**, **interpretability**, and **flexibility**.


This repository includes example data, a [minimal example python script](`run_topicmodels.py`), a [minimal example jupyter notebook](run_topicmodels.py) and a [Dockerfile](Dockerfile) that launches JupyterLab so you can interactively run the examples.



## Contents
- run_topicmodels.py — example script that demonstrates using the topicmodels factory and models
- data/10k_amazon.csv — example dataset included in the repo
- requirements.txt — pip installable dependencies
- pyproject.toml — optional Poetry config (if present)
- Dockerfile — builds an image and launches JupyterLab with the repository available in the container
- packages/ — package sources implementing the models

## Quick overview
- You can run the code locally with a virtualenv + pip, with Poetry, or inside Docker (recommended for reproducible environment).
- The Docker image starts JupyterLab and exposes your repository files and the example data inside the notebook environment.

### Setup — choose one

The package is soon available on PyPI.

As of now there are multiple options for installation available:

#### 1) Using pip + virtualenv (recommended)
```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

#### 2) Using Poetry
```bash
poetry install
poetry shell
```

#### 3) Using Docker (isolated, reproducible)
- Build image (run from repository root)
```bash
docker build -t topicmodels-jupyter .
```

- Run container and open JupyterLab on host port 8888
Windows (cmd):
```bash
docker run --rm -p 8888:8888 --name topicmodels_jupyter -v "%cd%":/workspace topicmodels-jupyter
```
PowerShell:
```powershell
docker run --rm -p 8888:8888 --name topicmodels_jupyter -v ${PWD}:/workspace topicmodels-jupyter
```
Linux / WSL:
```bash
docker run --rm -p 8888:8888 --name topicmodels_jupyter -v "$(pwd)":/workspace topicmodels-jupyter
```

Important notes for Docker
- The container runs JupyterLab on port 8888 inside the container. Map host port 8888 to container port 8888 (`-p 8888:8888`). Mapping a different host port is OK (e.g. `-p 8880:8888`) — then open `http://localhost:8880`.
- For local development you should mount your repo into the container using `-v` (see examples above) so edits persist.
- By default the provided Dockerfile disables the Jupyter token for local convenience. For any public or shared deployment, secure the server (set a token, password, or use a proxy).
- If JAX/jaxlib require platform-specific wheels (GPU vs CPU) you may need to adjust `requirements.txt` or the Dockerfile as described in JAX docs.

Troubleshooting
- If you cannot open the notebook URL:
  - Confirm the container is running: `docker ps`
  - View startup logs: `docker logs -f topicmodels_jupyter`
  - Look for the Jupyter server URL line (should bind to 0.0.0.0:8888) and any error messages from pip/jupyter startup.
  - Common mistake: wrong port mapping (e.g. `-p 888:888` instead of `-p 8888:8888`).
- If the container exits immediately, check `docker ps -a` and `docker logs <container>` for errors.

Using the repository in JupyterLab
- After opening JupyterLab (http://localhost:8888), the `/workspace` directory will show the project files including `run_topicmodels.py` and `data/10k_amazon.csv`.
- To run the example script in a notebook cell:
```python
# Option A: run the script top-to-bottom (good for quick demo)
%run run_topicmodels.py
```
- Or open a new Python notebook and selectively import functions from your package code (recommended once you refactor `run_topicmodels.py` to expose functions instead of running top-level code).
- The example script demonstrates:
  - loading `data/10k_amazon.csv`
  - building a document-term matrix with sklearn CountVectorizer
  - creating models via `topicmodels()` factory
  - running `train_step(...)` and inspecting results (`return_topics()`, `return_beta()`, `return_top_words_per_topic()`)

Run the example as a script (local/python env)
```bash
python run_topicmodels.py
```
Note: `run_topicmodels.py` runs substantial computations and may require appropriate CPU/GPU resources and compatible JAX wheels.

Recommended workflow
- Use Docker for a reproducible environment that includes JupyterLab and example data.
- Mount your repo to persist edits and to iterate inside notebooks.
- For development, refactor long example scripts to small unit-testable functions and import them in notebooks.

Security and production notes
- The Dockerfile disables the Jupyter token for ease of local development. Do not use this in production without adding authentication.
- For GPU support with JAX, follow the JAX installation guide and use a CUDA-capable base image and host drivers.

## License & Contributing
- Add your license text here.
- Pull requests and issues welcome. For code changes, include tests and documentation.













FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# ENV UV_COMPILE_BYTECODE=1
# ENV UV_LINK_MODE=copy
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"
ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

WORKDIR /app
COPY pyproject.toml /app
COPY uv.lock /app
COPY README.md /app
COPY src /app/src
RUN uv venv --python 3.10 /opt/venv
RUN uv sync --active
RUN uv add --active -U "jax[cuda12]"
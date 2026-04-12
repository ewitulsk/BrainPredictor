FROM runpod/base:1.0.3-cuda1281-ubuntu2204

# System dependencies for video/audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Phase 1: PyTorch with CUDA 12.8 — supports both H100 (sm_90) and Blackwell (sm_120)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Phase 2: Install tribev2 with --no-deps to bypass its stale torch<2.7 pin
RUN pip install --no-cache-dir --no-deps \
    "tribev2 @ git+https://github.com/facebookresearch/tribev2.git"

# Phase 3: Install tribev2's remaining deps + our requirements
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install uv — tribev2 invokes WhisperX via `uvx whisperx ...` for transcription
RUN pip install --no-cache-dir uv

# Ensure python3 points to the same interpreter pip uses
RUN ln -sf $(which python3) /usr/bin/python3 || true

# Download spaCy English model
RUN python3 -m spacy download en_core_web_sm

# Copy handler code
COPY src/ /src/
WORKDIR /src

# All caches live on the RunPod Network Volume (mounted at /runpod-volume in serverless).
# Persists across cold starts — HF models, torch, uv tools, and WhisperX all survive restarts.
ENV XDG_CACHE_HOME=/runpod-volume/.cache
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV TORCH_HOME=/runpod-volume/.cache/torch
ENV PIP_CACHE_DIR=/runpod-volume/.cache/pip
ENV CACHE_FOLDER=/runpod-volume/.cache/tribev2

# uv's tool and wheel caches on the Network Volume — WhisperX gets installed here
# on first cold start (~10 GB) and reused on every subsequent start.
ENV UV_TOOL_DIR=/runpod-volume/.cache/uv-tools
ENV UV_CACHE_DIR=/runpod-volume/.cache/uv-cache

# runpod/base sets HF_HUB_ENABLE_HF_TRANSFER=1 by default, but the isolated
# `uvx whisperx` env that tribev2 spawns doesn't include the hf_transfer package,
# so downloads crash. Force it off container-wide.
ENV HF_HUB_ENABLE_HF_TRANSFER=0

CMD ["python3", "-u", "/src/handler.py"]

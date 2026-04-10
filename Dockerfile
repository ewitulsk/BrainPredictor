FROM runpod/base:0.6.3-cuda12.1.0

# System dependencies for video/audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# pip installs into python3.13 on this base image
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install uv — tribev2 invokes WhisperX via `uvx whisperx ...` for transcription
RUN pip install --no-cache-dir uv

# Ensure python3 points to the same interpreter pip uses
RUN ln -sf $(which python3.13) /usr/bin/python3

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

CMD ["python3", "-u", "/src/handler.py"]

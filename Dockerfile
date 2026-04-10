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

# Pre-download TRIBE v2 checkpoint (~709MB) into the image
# This avoids re-downloading on every cold start
RUN python3 -c "\
from tribev2.demo_utils import TribeModel; \
TribeModel.from_pretrained('facebook/tribev2', cache_folder='/baked_models')"

# Copy handler code
COPY src/ /src/
WORKDIR /src

# Cache directories — use /cache for runtime model downloads (Network Volume mount point)
# /baked_models contains the pre-downloaded TRIBE checkpoint
ENV HF_HOME=/cache/huggingface
ENV TORCH_HOME=/cache/torch
ENV CACHE_FOLDER=/cache

# uv's tool and wheel caches on the Network Volume — WhisperX gets installed
# here on first use and reused on subsequent cold starts
ENV UV_TOOL_DIR=/cache/uv-tools
ENV UV_CACHE_DIR=/cache/uv-cache

CMD ["python3", "-u", "/src/handler.py"]

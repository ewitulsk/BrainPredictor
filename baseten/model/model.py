"""TRIBE v2 brain encoding model — Baseten Truss wrapper.

Ports src/handler.py + src/predict.py + src/storage.py from the RunPod setup
into the Baseten Model class lifecycle.
"""

import base64
import io
import logging
import os
import tempfile
import traceback
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Ensure uv's baked tool install is visible to the uvx whisperx subprocess.
os.environ.setdefault("UV_TOOL_DIR", "/opt/uv-tools")
os.environ.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")


class Model:
    def __init__(self, **kwargs):
        self._secrets = kwargs.get("secrets") or {}
        self._model = None

    def load(self):
        # Propagate HF token before any HF hub call so gated LLaMA 3.2 downloads work.
        token = self._secrets.get("hf_access_token")
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token

        # Block until model_cache prefetch finishes before touching the files.
        try:
            from truss.base import lazy_data_resolver
            lazy_data_resolver.block_until_download_complete()
        except Exception as e:
            logger.warning("lazy_data_resolver not available or failed: %s", e)

        from tribev2.demo_utils import TribeModel

        cache_folder = Path(os.environ.get("CACHE_FOLDER", "/app/model_cache/tribev2"))
        # Pass the LOCAL cache dir as checkpoint_dir so from_pretrained reads
        # config.yaml + best.ckpt straight off disk instead of re-downloading
        # via hf_hub_download (which hung silently on xet for 28+ min).
        checkpoint_dir = cache_folder
        logger.info(
            "Loading TRIBE v2 from checkpoint_dir=%s (contents: %s)",
            checkpoint_dir,
            sorted(p.name for p in checkpoint_dir.iterdir()) if checkpoint_dir.exists() else "MISSING",
        )
        logger.info("Calling TribeModel.from_pretrained...")
        self._model = TribeModel.from_pretrained(
            checkpoint_dir,
            cache_folder=cache_folder,
        )
        logger.info("TRIBE v2 loaded successfully")

    def predict(self, model_input: dict) -> dict:
        """Input (exactly one of video_url, audio_url, text required):
            video_url: URL to a video file (.mp4, .avi, .mkv, .mov, .webm)
            audio_url: URL to an audio file (.wav, .mp3, .flac, .ogg)
            text: text string to predict brain response for
            upload_url: optional presigned S3 PUT URL for uploading the .npy result
        """
        video_url = model_input.get("video_url")
        audio_url = model_input.get("audio_url")
        text = model_input.get("text")
        upload_url = model_input.get("upload_url")

        provided = sum(x is not None for x in [video_url, audio_url, text])
        if provided != 1:
            return {"error": "Provide exactly one of: video_url, audio_url, text"}

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                preds = self._run_prediction(
                    video_url=video_url,
                    audio_url=audio_url,
                    text=text,
                    tmpdir=tmpdir,
                )
                return self._serialize(preds, upload_url=upload_url)
            except Exception as e:
                logger.error("Prediction failed: %s", e, exc_info=True)
                return {"error": str(e), "traceback": traceback.format_exc()}

    # ----- helpers (ported from src/predict.py + src/storage.py) -----

    @staticmethod
    def _download(url: str, dest: str) -> None:
        logger.info("Downloading %s -> %s", url, dest)
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        logger.info("Downloaded %.1f MB", size_mb)

    def _run_prediction(self, *, video_url, audio_url, text, tmpdir) -> np.ndarray:
        if video_url:
            ext = os.path.splitext(video_url.split("?")[0])[1] or ".mp4"
            path = os.path.join(tmpdir, f"input{ext}")
            self._download(video_url, path)
            df = self._model.get_events_dataframe(video_path=path)
        elif audio_url:
            ext = os.path.splitext(audio_url.split("?")[0])[1] or ".wav"
            path = os.path.join(tmpdir, f"input{ext}")
            self._download(audio_url, path)
            df = self._model.get_events_dataframe(audio_path=path)
        else:
            text_path = os.path.join(tmpdir, "input.txt")
            Path(text_path).write_text(text)
            df = self._model.get_events_dataframe(text_path=text_path)

        logger.info("Events dataframe: %d rows", len(df))
        preds, _segments = self._model.predict(events=df)
        logger.info("Predictions shape: %s", preds.shape)
        return preds

    @staticmethod
    def _serialize(preds: np.ndarray, *, upload_url: str | None) -> dict:
        buf = io.BytesIO()
        np.save(buf, preds)
        npy = buf.getvalue()

        result = {
            "shape": list(preds.shape),
            "dtype": str(preds.dtype),
            "n_timesteps": int(preds.shape[0]),
            "n_vertices": int(preds.shape[1]),
        }
        if upload_url:
            logger.info("Uploading %d bytes to presigned URL", len(npy))
            r = requests.put(
                upload_url,
                data=npy,
                headers={"Content-Type": "application/octet-stream"},
                timeout=120,
            )
            r.raise_for_status()
            result["url"] = upload_url.split("?")[0]
        else:
            logger.info("Returning %d bytes inline as base64", len(npy))
            result["data_base64"] = base64.b64encode(npy).decode("ascii")
        return result

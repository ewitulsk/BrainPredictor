"""TRIBE v2 brain encoding model — Baseten Truss wrapper.

Ports src/handler.py + src/predict.py + src/storage.py from the RunPod setup
into the Baseten Model class lifecycle.
"""

import base64
import io
import logging
import os
import subprocess
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
        # Required by Baseten when model_cache is used — the runtime passes this
        # in kwargs and enforces that we call block_until_download_complete().
        self._lazy_data_resolver = kwargs.get("lazy_data_resolver")
        self._model = None

    def load(self):
        # Propagate HF token before any HF hub call so gated LLaMA 3.2 downloads work.
        token = self._secrets.get("hf_access_token")
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token

        # Block until model_cache prefetch finishes before touching the files.
        if self._lazy_data_resolver is not None:
            logger.info("Waiting for lazy_data_resolver to finish downloads...")
            self._lazy_data_resolver.block_until_download_complete()
            logger.info("lazy_data_resolver downloads complete")
        else:
            logger.warning("lazy_data_resolver kwarg not provided")

        # Diagnostic: verify the uvx whisperx subprocess venv actually sees CUDA.
        # If it doesn't, whisperx silently falls back to CPU (via ctranslate2) and
        # an 8-min video transcription takes 30+ min instead of ~30 seconds.
        self._verify_whisperx_cuda()

        # Monkey-patch tribev2's whisperx subprocess call to stream output instead
        # of buffering it, so we can actually see progress and errors in Baseten logs.
        self._patch_whisperx_streaming()

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

    # ----- whisperx GPU diagnostic + streaming monkey-patch -----

    @staticmethod
    def _verify_whisperx_cuda() -> None:
        """Run a tiny `uvx` subprocess to verify whisperx's isolated venv sees CUDA.

        tribev2 passes `--device cuda` based on the parent process's torch, but
        `uvx whisperx` runs in its own isolated venv with its own torch wheel.
        If that venv can't initialize CUDA, ctranslate2 silently falls back to
        CPU and transcription takes 30x longer.
        """
        try:
            result = subprocess.run(
                [
                    "uvx",
                    "--from", "whisperx",
                    "python",
                    "-c",
                    "import torch, ctranslate2; "
                    "print('torch_version:', torch.__version__); "
                    "print('cuda_available:', torch.cuda.is_available()); "
                    "print('cuda_device_count:', torch.cuda.device_count()); "
                    "print('ctranslate2_version:', ctranslate2.__version__); "
                    "print('ctranslate2_cuda_device_count:', ctranslate2.get_cuda_device_count())",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            logger.info("whisperx venv diagnostic stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("whisperx venv diagnostic stderr:\n%s", result.stderr)
            if result.returncode != 0:
                logger.error("whisperx venv diagnostic FAILED (rc=%d)", result.returncode)
        except Exception as e:
            logger.error("whisperx venv diagnostic raised: %s", e)

    @staticmethod
    def _patch_whisperx_streaming() -> None:
        """Replace tribev2's buffered subprocess.run with a streaming Popen.

        Upstream uses `subprocess.run(cmd, capture_output=True)`, which hides all
        output until the process finishes. On a stuck/slow run this leaves the
        Baseten logs completely blank. We monkey-patch the method to use Popen
        and forward stdout/stderr line-by-line to our logger in real time.
        """
        import json
        import shutil
        from pathlib import Path as _Path

        import pandas as pd
        import torch
        from tribev2 import eventstransforms

        orig_cls = eventstransforms.ExtractWordsFromAudio

        language_codes = dict(
            english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
        )

        def _streaming_get_transcript(wav_filename: _Path, language: str) -> pd.DataFrame:
            if language not in language_codes:
                raise ValueError(f"Language {language} not supported")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            with tempfile.TemporaryDirectory() as output_dir:
                logger.info(
                    "Running whisperx via uvx (device=%s, compute_type=%s, file=%s)",
                    device, compute_type, wav_filename,
                )
                cmd = [
                    "uvx", "whisperx",
                    str(wav_filename),
                    "--model", "large-v3",
                    "--language", language_codes[language],
                    "--device", device,
                    "--compute_type", compute_type,
                    "--batch_size", "16",
                ]
                if language == "english":
                    cmd += ["--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H"]
                cmd += [
                    "--output_dir", output_dir,
                    "--output_format", "json",
                ]

                env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
                env.setdefault("PYTHONUNBUFFERED", "1")

                logger.info("whisperx cmd: %s", " ".join(cmd))
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    logger.info("[whisperx] %s", line.rstrip())
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"whisperx failed with exit code {rc}")

                json_path = _Path(output_dir) / f"{wav_filename.stem}.json"
                transcript = json.loads(json_path.read_text())

            words = []
            for i, segment in enumerate(transcript["segments"]):
                sentence = segment["text"].replace('"', "")
                for word in segment["words"]:
                    if "start" not in word:
                        continue
                    words.append({
                        "text": word["word"].replace('"', ""),
                        "start": word["start"],
                        "duration": word["end"] - word["start"],
                        "sequence_id": i,
                        "sentence": sentence,
                    })
            return pd.DataFrame(words)

        orig_cls._get_transcript_from_audio = staticmethod(_streaming_get_transcript)
        logger.info("Patched tribev2.ExtractWordsFromAudio with streaming whisperx")

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

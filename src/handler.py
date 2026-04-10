import os
import tempfile
import traceback
import logging

import requests
import runpod

from predict import run_prediction
from storage import upload_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_DURATION_SECONDS = 600  # 10 minute input limit


def _download_file(url: str, dest_path: str) -> None:
    """Download a file from URL with streaming."""
    logger.info(f"Downloading {url} -> {dest_path}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    logger.info(f"Downloaded {size_mb:.1f} MB")


def handler(job: dict) -> dict:
    """RunPod serverless handler for TRIBE v2 brain predictions.

    Input (exactly one of video_url, audio_url, text required):
        video_url: URL to a video file (.mp4, .avi, .mkv, .mov, .webm)
        audio_url: URL to an audio file (.wav, .mp3, .flac, .ogg)
        text: Text string to predict brain response for
        upload_url: Optional presigned S3 PUT URL for uploading results
    """
    job_input = job["input"]

    video_url = job_input.get("video_url")
    audio_url = job_input.get("audio_url")
    text = job_input.get("text")
    upload_url = job_input.get("upload_url")

    inputs_provided = sum(x is not None for x in [video_url, audio_url, text])
    if inputs_provided != 1:
        return {"error": "Provide exactly one of: video_url, audio_url, text"}

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            video_path = audio_path = text_string = None

            if video_url:
                ext = os.path.splitext(video_url.split("?")[0])[1] or ".mp4"
                video_path = os.path.join(tmpdir, f"input{ext}")
                _download_file(video_url, video_path)

            elif audio_url:
                ext = os.path.splitext(audio_url.split("?")[0])[1] or ".wav"
                audio_path = os.path.join(tmpdir, f"input{ext}")
                _download_file(audio_url, audio_path)

            else:
                text_string = text

            preds = run_prediction(
                video_path=video_path,
                audio_path=audio_path,
                text_string=text_string,
                tmpdir=tmpdir,
            )

            result = upload_predictions(preds, upload_url=upload_url)
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})

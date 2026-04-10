import os
import logging
from pathlib import Path

import numpy as np

from tribev2.demo_utils import TribeModel

logger = logging.getLogger(__name__)

CACHE_FOLDER = Path(os.environ.get("CACHE_FOLDER", "/cache"))

logger.info("Loading TRIBE v2 model...")
model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder=CACHE_FOLDER,
)
logger.info("TRIBE v2 model loaded successfully.")


def run_prediction(
    video_path: str | None = None,
    audio_path: str | None = None,
    text_string: str | None = None,
    tmpdir: str = "/tmp",
) -> np.ndarray:
    """Run TRIBE v2 prediction on a single input.

    Accepts exactly one of video_path, audio_path, or text_string.
    Returns predictions array of shape (n_timesteps, 20484).
    """
    inputs_provided = sum(x is not None for x in [video_path, audio_path, text_string])
    if inputs_provided != 1:
        raise ValueError("Provide exactly one of: video_path, audio_path, text_string")

    if text_string is not None:
        text_path = Path(tmpdir) / "input.txt"
        text_path.write_text(text_string)
        df = model.get_events_dataframe(text_path=str(text_path))
    elif audio_path is not None:
        df = model.get_events_dataframe(audio_path=audio_path)
    else:
        df = model.get_events_dataframe(video_path=video_path)

    logger.info(f"Events dataframe has {len(df)} rows")
    preds, segments = model.predict(events=df)
    logger.info(f"Predictions shape: {preds.shape}")

    return preds

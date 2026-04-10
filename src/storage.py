import base64
import io
import logging

import numpy as np
import requests

logger = logging.getLogger(__name__)


def upload_predictions(
    preds: np.ndarray,
    upload_url: str | None = None,
) -> dict:
    """Serialize predictions and either upload to a presigned URL or return base64-encoded.

    Args:
        preds: Numpy array of shape (n_timesteps, n_vertices).
        upload_url: Optional presigned S3 PUT URL. If provided, uploads the .npy
                    file there and returns the URL. Otherwise returns base64 data inline.

    Returns:
        Dict with shape, dtype, and either "url" or "data_base64".
    """
    buf = io.BytesIO()
    np.save(buf, preds)
    npy_bytes = buf.getvalue()

    result = {
        "shape": list(preds.shape),
        "dtype": str(preds.dtype),
        "n_timesteps": int(preds.shape[0]),
        "n_vertices": int(preds.shape[1]),
    }

    if upload_url:
        logger.info(f"Uploading predictions ({len(npy_bytes)} bytes) to presigned URL")
        resp = requests.put(
            upload_url,
            data=npy_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=120,
        )
        resp.raise_for_status()
        # Strip query params (presigned signature) from the URL for the response
        result["url"] = upload_url.split("?")[0]
    else:
        logger.info(f"Returning predictions inline as base64 ({len(npy_bytes)} bytes)")
        result["data_base64"] = base64.b64encode(npy_bytes).decode("ascii")

    return result

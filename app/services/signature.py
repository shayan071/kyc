"""Signature verification with a SigNet-style Siamese CNN (writer-independent).

The network maps a signature image to a 2048-d embedding; two signatures are compared by
cosine similarity. Architecture and preprocessing follow the `sigver` project
(Hafemann et al.), so its pretrained `signet.pth` weights can be loaded directly.
Weights are not bundled: put the file at KYC_SIGNATURE_WEIGHTS.
"""

import base64
import io
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.config import Settings
from app.schemas import SignatureResult

CANVAS_SIZE = (952, 1360)  # rows, cols (GPDS convention used by SigNet)
IMG_SIZE = (170, 242)
INPUT_SIZE = (150, 220)


# --------------------------------------------------------------------------- preprocessing
def _to_gray(data: bytes) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L")
    return np.array(img, dtype=np.uint8)


def _normalize_on_canvas(img: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), 2)
    # Otsu on the blurred image: on clean two-tone images the raw histogram only has two
    # values and Otsu degenerates to threshold 0, which finds no ink.
    threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = blurred > threshold
    rows, cols = np.where(~binary)  # ink pixels
    if rows.size == 0:
        raise ValueError("No ink found in the signature image.")
    r_center = int(rows.mean() - rows.min())
    c_center = int(cols.mean() - cols.min())
    cropped = img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    max_rows, max_cols = CANVAS_SIZE
    # Shrink very large signatures so they fit the canvas around their centre of mass.
    scale = min(1.0, (max_rows * 0.95) / cropped.shape[0], (max_cols * 0.95) / cropped.shape[1])
    if scale < 1.0:
        cropped = cv2.resize(
            cropped, (int(cropped.shape[1] * scale), int(cropped.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        r_center, c_center = int(r_center * scale), int(c_center * scale)

    h, w = cropped.shape
    r_start = int(np.clip(max_rows // 2 - r_center, 0, max_rows - h))
    c_start = int(np.clip(max_cols // 2 - c_center, 0, max_cols - w))
    canvas = np.full((max_rows, max_cols), 255, dtype=np.uint8)
    canvas[r_start : r_start + h, c_start : c_start + w] = cropped
    canvas[canvas > threshold] = 255  # remove background noise
    return canvas


def _resize_keep_ratio(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    width_ratio = img.shape[1] / width
    height_ratio = img.shape[0] / height
    if width_ratio > height_ratio:
        new_h, new_w = height, int(round(img.shape[1] / height_ratio))
    else:
        new_w, new_h = width, int(round(img.shape[0] / width_ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return _crop_center(resized, size)


def _crop_center(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    r = (img.shape[0] - h) // 2
    c = (img.shape[1] - w) // 2
    return img[r : r + h, c : c + w]


def preprocess(gray: np.ndarray) -> np.ndarray:
    canvas = _normalize_on_canvas(gray)
    inverted = 255 - canvas  # background 0, strokes bright
    resized = _resize_keep_ratio(inverted, IMG_SIZE)
    return _crop_center(resized, INPUT_SIZE)


# --------------------------------------------------------------------------- model
def _build_signet():
    import torch.nn as nn

    def conv_bn_relu(cin, cout, k, stride=1, pad=0):
        return nn.Sequential(
            OrderedDict(
                conv=nn.Conv2d(cin, cout, k, stride, pad, bias=False),
                bn=nn.BatchNorm2d(cout, eps=1e-6, momentum=0.1),
                relu=nn.ReLU(),
            )
        )

    def linear_bn_relu(fin, fout):
        return nn.Sequential(
            OrderedDict(
                fc=nn.Linear(fin, fout, bias=False),
                bn=nn.BatchNorm1d(fout, eps=1e-6, momentum=0.1),
                relu=nn.ReLU(),
            )
        )

    class SigNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                OrderedDict(
                    conv1=conv_bn_relu(1, 96, 11, stride=4),
                    maxpool1=nn.MaxPool2d(3, 2),
                    conv2=conv_bn_relu(96, 256, 5, pad=2),
                    maxpool2=nn.MaxPool2d(3, 2),
                    conv3=conv_bn_relu(256, 384, 3, pad=1),
                    conv4=conv_bn_relu(384, 384, 3, pad=1),
                    conv5=conv_bn_relu(384, 256, 3, pad=1),
                    maxpool3=nn.MaxPool2d(3, 2),
                )
            )
            self.fc_layers = nn.Sequential(
                OrderedDict(
                    fc1=linear_bn_relu(256 * 3 * 5, 2048),
                    fc2=linear_bn_relu(2048, 2048),
                )
            )

        def forward(self, x):
            x = self.conv_layers(x)
            return self.fc_layers(x.view(x.shape[0], -1))

    return SigNet()


class SignatureService:
    def __init__(self, settings: Settings):
        self.s = settings
        self._model = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # -- loading ---------------------------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None or self._load_error:
            return
        with self._lock:
            if self._model is not None or self._load_error:
                return
            path = Path(self.s.signature_weights)
            if not path.exists():
                self._load_error = (
                    f"Signature weights not found at '{path}'. Download SigNet weights "
                    "(sigver 'signet.pth') and set KYC_SIGNATURE_WEIGHTS."
                )
                return
            try:
                import torch

                model = _build_signet()
                state = torch.load(path, map_location="cpu")
                if isinstance(state, (tuple, list)):
                    state = state[0]
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state)
                model.eval()
                self._model = model
            except Exception as exc:
                self._load_error = f"Could not load signature model: {exc}"

    @property
    def status(self) -> str | None:
        """None when the model is ready, otherwise the reason it is not."""
        self._ensure_model()
        return self._load_error

    # -- inference -------------------------------------------------------------------------
    def _embed(self, processed: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(processed).float().div(255).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return self._model(x).squeeze(0).numpy()

    def crop_signature_from_card(self, card_bytes: bytes) -> np.ndarray:
        gray = _to_gray(card_bytes)
        h, w = gray.shape
        x0, y0, x1, y1 = self.s.roi
        return gray[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]

    @staticmethod
    def _png_b64(gray: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(gray).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def verify(self, card_bytes: bytes, specimen_bytes: bytes) -> SignatureResult:
        threshold = self.s.signature_threshold
        err = self.status
        if err:
            return SignatureResult(verdict="unavailable", threshold=threshold, message=err)

        try:
            card_crop = self.crop_signature_from_card(card_bytes)
            crop_png = self._png_b64(card_crop)
            specimen = _to_gray(specimen_bytes)
        except Exception as exc:
            return SignatureResult(
                verdict="unavailable", threshold=threshold, message=f"Could not read images: {exc}"
            )

        try:
            e1 = self._embed(preprocess(card_crop))
            e2 = self._embed(preprocess(specimen))
        except ValueError as exc:
            return SignatureResult(
                verdict="unavailable",
                threshold=threshold,
                message=str(exc),
                cropped_signature_png_b64=crop_png,
            )

        score = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))
        if score >= threshold:
            verdict = "genuine"
        elif score < threshold - self.s.signature_margin:
            verdict = "forged"
        else:
            verdict = "uncertain"
        return SignatureResult(
            verdict=verdict,
            score=round(score, 4),
            threshold=threshold,
            cropped_signature_png_b64=crop_png,
        )

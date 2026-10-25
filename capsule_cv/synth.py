"""Synthetic capsule-endoscopy video with ground truth, for testing the pipeline without patient data.

Simulates: three GI regions with distinct mucosa colour/texture, a camera that pans with dwell periods
(near-duplicate runs), blur / bubbles / bile-debris / over-exposure artefacts, bleeding episodes, and
focal lesions (pale ulcer-like and red angioectasia-like spots).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .ingest import Frame

# BGR mucosa base colours per region
REGION_BASE = {"stomach": (105, 118, 200), "small bowel": (80, 150, 215), "colon": (150, 168, 205)}


def _texture(rng: np.random.Generator, size: int, villi: bool) -> np.ndarray:
    low = cv2.resize(rng.normal(0, 1, (size // 32, size // 32)).astype(np.float32), (size, size),
                     interpolation=cv2.INTER_CUBIC)
    mid = cv2.resize(rng.normal(0, 1, (size // 8, size // 8)).astype(np.float32), (size, size),
                     interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.0)
    tex = 0.6 * low + 0.3 * mid + 0.5 * fine  # fine = surface pits / microvasculature
    if villi:
        tex += 0.6 * cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.6)
    yy = np.arange(size, dtype=np.float32)[:, None]
    tex += 0.5 * np.sin(yy / 23.0 + 2 * low)  # mucosal folds
    return tex


class SyntheticCapsule:
    def __init__(self, n_frames: int = 600, size: int = 256, fps: float = 4.0, seed: int = 7):
        self.n, self.size, self.fps = n_frames, size, fps
        self.rng = np.random.default_rng(seed)
        b1, b2 = int(0.15 * n_frames), int(0.75 * n_frames)
        self.boundaries = [b1, b2]
        self.tex = {name: _texture(self.rng, 1024, name == "small bowel") for name in REGION_BASE}
        span = lambda lo, hi, ln: (int(lo * n_frames), int(lo * n_frames) + ln)
        self.truth = {
            "bleeding": [span(0.35, 0, 18), span(0.62, 0, 12)],
            "lesion_pale": [span(0.48, 0, 10)],
            "lesion_red": [span(0.55, 0, 8)],
            "blur": [span(0.22, 0, 10), span(0.85, 0, 8)],
            "bubbles": [span(0.42, 0, 10)],
            "debris": [span(0.70, 0, 10)],
            "dwell": [span(0.28, 0, 20), span(0.80, 0, 15)],
        }
        yy, xx = np.mgrid[:size, :size].astype(np.float32)
        r = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
        self.vignette = np.clip(1.15 - 0.55 * r ** 2, 0, 1)[..., None]
        self.circle = (r <= 0.94)[..., None]

    def region(self, i: int) -> str:
        return "stomach" if i < self.boundaries[0] else "small bowel" if i < self.boundaries[1] else "colon"

    def _in(self, kind: str, i: int) -> bool:
        return any(a <= i < b for a, b in self.truth[kind])

    def frames(self) -> Iterator[Frame]:
        s, rng = self.size, self.rng
        pos = np.array([300.0, 300.0])
        for i in range(self.n):
            if not self._in("dwell", i):
                pos = pos + rng.normal(0, 12, 2) + [6, 5]
                pos = np.where((pos < 0) | (pos > 1024 - s - 1), rng.uniform(0, 1024 - s - 1, 2), pos)
            name = self.region(i)
            x, y = pos.astype(int)
            tex = self.tex[name][y:y + s, x:x + s]
            base = np.array(REGION_BASE[name], np.float32)
            img = base * (1 + 0.15 * tex[..., None])
            # dark lumen
            cx, cy = s // 2 + int(30 * np.sin(i / 17)), s // 2 + int(30 * np.cos(i / 13))
            yy, xx = np.mgrid[:s, :s]
            img *= 1 - 0.75 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (s / 9) ** 2))[..., None]
            img = img * self.vignette
            if not self._in("dwell", i):
                img += rng.normal(0, 2.0, img.shape)  # sensor noise; dwell frames stay near-identical
            img = np.clip(img, 0, 255).astype(np.uint8)

            if self._in("bleeding", i):
                k = i - next(a for a, b in self.truth["bleeding"] if a <= i < b)
                blood = np.zeros((s, s), np.uint8)
                cv2.ellipse(blood, (s // 3 + 2 * k, s // 2), (28 + 2 * k, 20 + k), 15, 0, 360, 255, -1)
                blood = cv2.GaussianBlur(blood, (0, 0), 3) > 120
                img[blood] = (np.array([22, 14, 150]) + rng.normal(0, 6, (blood.sum(), 3))).clip(0, 255)
            if self._in("lesion_pale", i):
                p = (int(0.65 * s), int(0.35 * s))
                cv2.circle(img, p, 15, (60, 60, 190), -1)  # inflamed rim
                cv2.circle(img, p, 11, (170, 200, 222), -1)  # fibrin base
            if self._in("lesion_red", i):
                cv2.circle(img, (int(0.3 * s), int(0.65 * s)), 7, (40, 30, 230), -1)
            if self._in("bubbles", i):
                for _ in range(40):
                    c = tuple(int(v) for v in rng.integers(20, s - 20, 2))
                    cv2.circle(img, c, int(rng.integers(5, 16)), (235, 240, 240), 1, cv2.LINE_AA)
            if self._in("debris", i):
                for _ in range(12):
                    c = tuple(int(v) for v in rng.integers(30, s - 30, 2))
                    cv2.circle(img, c, int(rng.integers(15, 35)), (40, 140, 110), -1)
            if self._in("blur", i):
                img = cv2.GaussianBlur(img, (0, 0), 4)
            img[~self.circle[..., 0]] = 0
            yield Frame(i, i / self.fps, img)

    def ground_truth(self) -> dict:
        return {"fps": self.fps, "n_frames": self.n, "size": self.size,
                "region_boundaries_frame": self.boundaries,
                "region_boundaries_s": [b / self.fps for b in self.boundaries],
                "segments_frame": {k: [list(x) for x in v] for k, v in self.truth.items()}}


def write_video(out: str | Path, n_frames: int = 600, size: int = 256, fps: float = 4.0, seed: int = 7,
                as_images: bool = False) -> Path:
    """Write an .mp4 (or a folder of PNGs) plus <name>.truth.json; returns the media path."""
    out = Path(out)
    cap = SyntheticCapsule(n_frames, size, fps, seed)
    if as_images:
        out.mkdir(parents=True, exist_ok=True)
        for f in cap.frames():
            cv2.imwrite(str(out / f"frame_{f.index:06d}.png"), f.image)
        truth_path = out / "truth.json"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps, (size, size))
        for f in cap.frames():
            vw.write(f.image)
        vw.release()
        truth_path = out.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(cap.ground_truth(), indent=2))
    return out

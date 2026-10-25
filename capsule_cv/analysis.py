"""Per-frame computer vision for capsule endoscopy, plus study-level aggregation.

Per frame: image quality (brightness, sharpness, over-exposure, bile/debris), suspected blood
(red-pixel fraction), focal lesion candidates (pale = ulcer/fibrin-like, red = angioectasia-like),
and redundancy against the last retained frame. Per study: events (runs of flagged frames),
GI-region segmentation with transit times, and reading-time reduction from redundancy removal.

These are classical colour/texture heuristics meant as triage aids for a human reader, not a diagnosis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

import cv2
import numpy as np

from .ingest import Frame

REGION_NAMES = ("stomach", "small bowel", "colon")
THUMB_MAX = 320
MAX_ANNOTATED = 3000
OVERVIEW_TARGET = 200


@dataclass
class Thresholds:
    min_brightness: float = 35.0
    max_brightness: float = 225.0
    min_sharpness: float = 0.44  # Crete-Roffet no-reference score, 0 = fully blurred, 1 = sharp
    max_overexposed: float = 0.20
    max_debris: float = 0.45
    bleed_fraction: float = 0.03  # fraction of field-of-view pixels that look like blood
    bleed_red_ratio: float = 2.2  # R/G above this separates blood from pink mucosa
    lesion_min_area: float = 0.0015
    lesion_max_area: float = 0.20
    pale_delta_l: float = 28.0  # L* above mucosa median
    red_delta_a: float = 20.0  # a* above mucosa median
    duplicate_hist_corr: float = 0.98
    duplicate_mad: float = 5.0
    event_gap_s: float = 3.0


@dataclass
class FrameResult:
    index: int
    timestamp: float
    brightness: float
    sharpness: float
    overexposed: float
    debris: float
    usable: bool
    bleeding_score: float
    bleeding: bool
    lesions: list[dict] = field(default_factory=list)
    duplicate: bool = False
    mean_a: float = 0.0
    mean_b: float = 0.0
    region: str = ""


@dataclass
class Event:
    kind: str
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    peak_index: int
    peak_score: float
    n_frames: int
    region: str = ""


def field_of_view_mask(img: np.ndarray) -> np.ndarray:
    """Capsule images are usually a circular field on black; fall back to the full frame otherwise."""
    h, w = img.shape[:2]
    c = max(4, min(h, w) // 16)
    corners = [img[:c, :c], img[:c, -c:], img[-c:, :c], img[-c:, -c:]]
    mask = np.full((h, w), 255, np.uint8)
    if np.mean([x.mean() for x in corners]) < 20:
        mask[:] = 0
        cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.47), 255, -1)
    return mask


def sharpness_score(gray: np.ndarray, region: np.ndarray) -> float:
    """No-reference sharpness (Crete-Roffet et al., 2007): share of edge strength lost when the frame is
    re-blurred. Already-blurred frames lose little. Largely independent of scene contrast, unlike
    Laplacian variance, which jumps when a bright lesion or bubble enters the view. Measured at half
    resolution so JPEG/MPEG block edges don't make a blurred frame look sharp."""
    g = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
    region = region[::2, ::2][:g.shape[0], :g.shape[1]]
    b = cv2.blur(cv2.blur(g, (9, 1)), (1, 9))
    blur = 0.0
    for ax, m in ((0, region[1:, :]), (1, region[:, 1:])):
        d, db = np.abs(np.diff(g, axis=ax)), np.abs(np.diff(b, axis=ax))
        total = float(d[m].sum())
        kept = float(np.maximum(0, d - db)[m].sum())
        blur = max(blur, (total - kept) / total if total > 0 else 1.0)
    return 1.0 - blur


def _thumb(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = THUMB_MAX / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img.copy()


def annotate(img: np.ndarray, r: FrameResult, red_mask: np.ndarray | None) -> np.ndarray:
    out = img.copy()
    if red_mask is not None and r.bleeding:
        tint = np.zeros_like(out)
        tint[:] = (255, 255, 0)
        sel = red_mask > 0
        out[sel] = cv2.addWeighted(out, 0.4, tint, 0.6, 0)[sel]
    for les in r.lesions:
        x, y, w, h = les["box"]
        color = (255, 255, 255) if les["kind"] == "pale" else (0, 255, 255)
        cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), color, 2)
    out = _thumb(out)
    tags = []
    if r.bleeding:
        tags.append(f"BLOOD {r.bleeding_score:.0%}")
    if r.lesions:
        tags.append("LESION " + ",".join(sorted({l["kind"] for l in r.lesions})))
    label = f"#{r.index} t={r.timestamp:.1f}s " + " ".join(tags)
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class FrameAnalyzer:
    def __init__(self, thresholds: Thresholds | None = None):
        self.th = thresholds or Thresholds()
        self._masks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._last_kept: tuple[np.ndarray, np.ndarray] | None = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _mask(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = img.shape[:2]
        if key not in self._masks:
            m = field_of_view_mask(img)
            self._masks[key] = (m, cv2.erode(m, np.ones((9, 9), np.uint8)))
        return self._masks[key]

    def analyze(self, frame: Frame) -> tuple[FrameResult, np.ndarray]:
        th, img = self.th, frame.image
        mask, inner = self._mask(img)
        fov = mask > 0
        n = max(int(fov.sum()), 1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        brightness = float(gray[fov].mean())
        sharpness = sharpness_score(gray, inner > 0)
        specular = (v > 240) & (s < 30)
        overexposed = float((specular & fov).sum() / n)
        debris_mask = (h >= 28) & (h <= 60) & (s > 80) & (v > 40) & fov
        debris = float(debris_mask.sum() / n)
        usable = (th.min_brightness <= brightness <= th.max_brightness and sharpness >= th.min_sharpness
                  and overexposed <= th.max_overexposed and debris <= th.max_debris)

        # Suspected blood: saturated red hue AND strong R/G dominance (pink mucosa fails the ratio).
        b, g, r = (img[..., i].astype(np.float32) for i in range(3))
        red_hue = (h <= 8) | (h >= 170)
        red = red_hue & (s >= 120) & (v >= 40) & (v <= 235) & (r / (g + 1.0) >= th.bleed_red_ratio) & fov
        red_mask = cv2.morphologyEx(red.astype(np.uint8) * 255, cv2.MORPH_OPEN, self._kernel)
        bleeding_score = float((red_mask > 0).sum() / n)

        # Lesion candidates: focal colour departures from the *local* mucosa background. A median filter
        # keeps edges, so bright tissue beside the dark lumen or a fold isn't mistaken for a spot.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        ksize = max(15, (min(img.shape[:2]) // 8) | 1)
        L, A = (lab[..., i].astype(np.float32) for i in (0, 1))
        dL = L - cv2.medianBlur(lab[..., 0], ksize).astype(np.float32)
        dA = A - cv2.medianBlur(lab[..., 1], ksize).astype(np.float32)
        tissue = fov & (L > 40) & ~specular
        mA, mB = (float(np.median(lab[..., i][tissue if tissue.any() else fov])) for i in (1, 2))
        lesions = []
        candidates = {
            "pale": (dL > th.pale_delta_l) & (s >= 30) & ~specular,
            # on frames that already read as bleeding, the blood itself is not a separate focal lesion
            "red": (dA > th.red_delta_a) & ~((red_mask > 0) & (bleeding_score >= th.bleed_fraction)),
        }
        # Tissue bordering blood or bile/debris reads as locally pale/red; don't call it a lesion.
        confounders = debris_mask | ((red_mask > 0) & (bleeding_score >= th.bleed_fraction))
        near_conf = cv2.dilate(confounders.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        for kind, cand in candidates.items():
            cand = cand & (inner > 0) & ~near_conf
            cm = cv2.morphologyEx(cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, self._kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
            for i in range(1, count):
                x, y, w, hh, area = (int(v_) for v_ in stats[i])
                frac = area / n
                solidity = area / max(w * hh, 1)  # spots are compact; fold/edge slivers are not
                if th.lesion_min_area <= frac <= th.lesion_max_area and solidity >= 0.5 and max(w, hh) <= 3 * min(w, hh):
                    lesions.append({"kind": kind, "box": [x, y, w, hh], "area": round(frac, 5)})

        # Redundancy vs last retained frame (capsules dwell, producing near-identical runs).
        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        duplicate = False
        if self._last_kept is not None:
            prev_small, prev_hist = self._last_kept
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            mad = float(np.abs(small - prev_small).mean())
            duplicate = corr >= th.duplicate_hist_corr and mad <= th.duplicate_mad
        if not duplicate:
            self._last_kept = (small, hist)

        result = FrameResult(
            index=frame.index, timestamp=frame.timestamp, brightness=round(brightness, 2),
            sharpness=round(sharpness, 3), overexposed=round(overexposed, 4), debris=round(debris, 4),
            usable=bool(usable), bleeding_score=round(bleeding_score, 4),
            bleeding=bleeding_score >= th.bleed_fraction, lesions=lesions if usable else [],
            duplicate=duplicate, mean_a=round(mA, 2), mean_b=round(mB, 2),
        )
        return result, red_mask


@dataclass
class Study:
    source: str
    results: list[FrameResult]
    events: list[Event]
    boundaries: list[float]  # timestamps of stomach->small bowel and small bowel->colon transitions
    region_confidence: float
    annotated: dict[int, np.ndarray]  # flagged frames, annotated thumbnails
    overview: dict[int, np.ndarray]  # evenly spaced thumbnails across the study

    def summary(self) -> dict:
        rs = self.results
        n = len(rs)
        duration = rs[-1].timestamp - rs[0].timestamp if n > 1 else 0.0
        usable = sum(r.usable for r in rs)
        dup = sum(r.duplicate for r in rs)
        to_read = sum(1 for r in rs if r.usable and not r.duplicate) + sum(
            1 for r in rs if not r.usable and (r.bleeding or r.lesions))
        out = {
            "source": self.source,
            "frames": n,
            "duration_s": round(duration, 1),
            "usable_frames": usable,
            "usable_pct": round(100 * usable / n, 1) if n else 0,
            "duplicate_frames": dup,
            "frames_to_review": to_read,
            "review_reduction_pct": round(100 * (1 - to_read / n), 1) if n else 0,
            "bleeding_frames": sum(r.bleeding for r in rs),
            "lesion_frames": sum(bool(r.lesions) for r in rs),
            "events": {k: sum(e.kind == k for e in self.events) for k in ("bleeding", "lesion")},
            "region_confidence": round(self.region_confidence, 2),
        }
        if len(self.boundaries) == 2 and n:
            t0 = rs[0].timestamp
            out["gastric_transit_s"] = round(self.boundaries[0] - t0, 1)
            out["small_bowel_transit_s"] = round(self.boundaries[1] - self.boundaries[0], 1)
        return out

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "boundaries": self.boundaries,
                "events": [asdict(e) for e in self.events]}


def _seg_cost(cs: np.ndarray, cs2: np.ndarray, a, b) -> np.ndarray:
    """SSE of segments [a, b) (vectorised over a and/or b), from cumulative sums with a leading zero row."""
    cnt = (np.asarray(b) - np.asarray(a)).astype(np.float64)[..., None]
    s = cs[b] - cs[a]
    return ((cs2[b] - cs2[a]) - s * s / np.maximum(cnt, 1)).sum(-1)


def segment_regions(results: list[FrameResult], max_bins: int = 1500) -> tuple[list[float], float]:
    """Two change points in mean colour/texture -> stomach / small bowel / colon boundaries.

    Returns (boundary timestamps, confidence = fraction of variance explained by the 3-segment fit).
    """
    pts = [r for r in results if r.usable and not r.duplicate]
    if len(pts) < 30:
        return [], 0.0
    feats = np.array([[r.mean_a, r.mean_b, r.sharpness] for r in pts], np.float64)
    times = np.array([r.timestamp for r in pts])
    bins = np.array_split(np.arange(len(pts)), min(max_bins, len(pts)))
    X = np.array([feats[ix].mean(0) for ix in bins])
    bin_t = np.array([times[ix[0]] for ix in bins])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.array([np.median(X[max(0, i - 2):i + 3], axis=0) for i in range(len(X))])  # despike

    m = len(X)
    cs = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, 0)])
    cs2 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X * X, 0)])
    min_len = max(3, int(0.05 * m))
    idx = np.arange(m + 1)
    head = _seg_cost(cs, cs2, 0, idx)  # cost of [0, i)
    tail = _seg_cost(cs, cs2, idx, np.full(m + 1, m))  # cost of [j, m)
    total = float(head[m])
    best = (np.inf, 0, 0)
    for i in range(min_len, m - 2 * min_len + 1):
        js = np.arange(i + min_len, m - min_len + 1)
        if not len(js):
            continue
        cost = head[i] + _seg_cost(cs, cs2, i, js) + tail[js]
        k = int(np.argmin(cost))
        if cost[k] < best[0]:
            best = (float(cost[k]), i, int(js[k]))
    if not np.isfinite(best[0]):
        return [], 0.0
    confidence = 1 - best[0] / total if total > 0 else 0.0
    return [float(bin_t[best[1]]), float(bin_t[best[2]])], float(confidence)


def group_events(results: list[FrameResult], kind: str, gap_s: float, min_frames: int = 2) -> list[Event]:
    if kind == "bleeding":
        flag, score = (lambda r: r.bleeding), (lambda r: r.bleeding_score)
    else:
        flag, score = (lambda r: bool(r.lesions)), (lambda r: max(l["area"] for l in r.lesions))
    events, run = [], []

    def close():
        if len(run) >= min_frames:
            peak = max(run, key=score)
            events.append(Event(kind, run[0].index, run[-1].index, run[0].timestamp, run[-1].timestamp,
                                peak.index, round(score(peak), 4), len(run), peak.region))

    for r in results:
        if flag(r):
            if run and r.timestamp - run[-1].timestamp > gap_s:
                close()
                run = []
            run.append(r)
    if run:
        close()
    return events


def region_at(t: float, boundaries: list[float]) -> str:
    if len(boundaries) != 2:
        return "unknown"
    return REGION_NAMES[0] if t < boundaries[0] else REGION_NAMES[1] if t < boundaries[1] else REGION_NAMES[2]


def analyze_stream(frames: Iterable[Frame], source: str = "", thresholds: Thresholds | None = None,
                   progress_every: int = 0) -> Study:
    analyzer = FrameAnalyzer(thresholds)
    results: list[FrameResult] = []
    annotated: dict[int, np.ndarray] = {}
    overview: dict[int, np.ndarray] = {}
    stride = 1
    for f in frames:
        r, red_mask = analyzer.analyze(f)
        results.append(r)
        if (r.bleeding or r.lesions) and len(annotated) < MAX_ANNOTATED:
            annotated[r.index] = annotate(f.image, r, red_mask)
        if r.usable and r.index % stride == 0:
            overview[r.index] = _thumb(f.image)
            if len(overview) > 2 * OVERVIEW_TARGET:  # decimate to keep coverage even with bounded memory
                stride *= 2
                overview = {k: v for k, v in overview.items() if k % stride == 0}
        if progress_every and len(results) % progress_every == 0:
            print(f"  analysed {len(results)} frames", flush=True)

    th = analyzer.th
    boundaries, conf = segment_regions(results)
    for r in results:
        r.region = region_at(r.timestamp, boundaries)
    events = group_events(results, "bleeding", th.event_gap_s) + group_events(results, "lesion", th.event_gap_s)
    events.sort(key=lambda e: e.start_time)
    return Study(source, results, events, boundaries, conf, annotated, overview)

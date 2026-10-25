"""Outputs: frames.csv, study.json, a self-contained report.html, and a highlights.mp4 of flagged frames."""
from __future__ import annotations

import base64
import csv
import html
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .analysis import REGION_NAMES, Study

REGION_COLORS = {"stomach": "#f4c7b8", "small bowel": "#f6e3a8", "colon": "#c9dcc0", "unknown": "#e2e2e2"}
VIDEO_SIZE = 320


def _b64jpg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _fmt_t(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def write_frames_csv(study: Study, path: Path) -> None:
    cols = ["index", "timestamp", "region", "usable", "duplicate", "brightness", "sharpness", "overexposed",
            "debris", "bleeding_score", "bleeding", "n_lesions", "lesion_kinds"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in study.results:
            w.writerow([r.index, f"{r.timestamp:.3f}", r.region, int(r.usable), int(r.duplicate), r.brightness,
                        r.sharpness, r.overexposed, r.debris, r.bleeding_score, int(r.bleeding), len(r.lesions),
                        "|".join(sorted({l["kind"] for l in r.lesions}))])


def _timeline_svg(study: Study, width: int = 1000, height: int = 170) -> str:
    rs = study.results
    if len(rs) < 2:
        return ""
    t0, t1 = rs[0].timestamp, rs[-1].timestamp
    span = max(t1 - t0, 1e-9)
    x = lambda t: 40 + (t - t0) / span * (width - 50)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="timeline" role="img" aria-label="Study timeline">']
    edges = [t0] + list(study.boundaries) + [t1]
    names = list(REGION_NAMES) if len(study.boundaries) == 2 else ["unknown"]
    for name, a, b in zip(names, edges, edges[1:]):
        parts.append(f'<rect x="{x(a):.1f}" y="10" width="{x(b) - x(a):.1f}" height="110" '
                     f'fill="{REGION_COLORS[name]}"><title>{name}</title></rect>')
        parts.append(f'<text x="{(x(a) + x(b)) / 2:.1f}" y="24" class="lbl" text-anchor="middle">{name}</text>')
    # bleeding score: max per pixel column so short events stay visible
    nb = width - 50
    cols = np.zeros(nb)
    for r in rs:
        c = min(nb - 1, int((r.timestamp - t0) / span * (nb - 1)))
        cols[c] = max(cols[c], r.bleeding_score)
    ymax = max(0.1, float(cols.max()))
    pts = " ".join(f"{40 + i:.0f},{120 - v / ymax * 90:.1f}" for i, v in enumerate(cols))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#b3122e" stroke-width="1.2"/>')
    for r in rs:
        if not r.usable:
            parts.append(f'<line x1="{x(r.timestamp):.1f}" x2="{x(r.timestamp):.1f}" y1="124" y2="132" '
                         'stroke="#888" stroke-width="0.6"/>')
    for e in study.events:
        color = "#b3122e" if e.kind == "bleeding" else "#1d5fa8"
        parts.append(f'<rect x="{x(e.start_time):.1f}" y="138" width="{max(2, x(e.end_time) - x(e.start_time)):.1f}" '
                     f'height="10" fill="{color}"><title>{e.kind} {_fmt_t(e.start_time)}-{_fmt_t(e.end_time)}'
                     f'</title></rect>')
    parts.append(f'<text x="4" y="40" class="lbl">blood</text><text x="4" y="131" class="lbl">poor</text>'
                 f'<text x="4" y="147" class="lbl">events</text>')
    parts.append(f'<text x="40" y="165" class="lbl">{_fmt_t(t0)}</text>'
                 f'<text x="{width - 10}" y="165" class="lbl" text-anchor="end">{_fmt_t(t1)}</text></svg>')
    return "".join(parts)


CSS = """
:root{--bg:#fafafa;--fg:#1b1b1b;--muted:#666;--card:#fff;--line:#ddd}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--muted:#aaa;--card:#1f1f1f;--line:#333}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1060px;margin:auto;padding:16px}
h1{font-size:22px;margin:4px 0}h2{font-size:17px;margin:26px 0 8px}
.warn{border-left:4px solid #c77700;background:rgba(199,119,0,.1);padding:8px 12px;margin:12px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.kpi b{display:block;font-size:20px}.kpi span{color:var(--muted);font-size:12px}
.timeline{width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px}
.lbl{font-size:10px;fill:#555}
table{border-collapse:collapse;width:100%;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.grid img{width:100%;border-radius:6px;display:block}
.grid figure{margin:0}.grid figcaption{font-size:11px;color:var(--muted)}
.tag{display:inline-block;padding:1px 7px;border-radius:10px;color:#fff;font-size:12px}
.bleeding{background:#b3122e}.lesion{background:#1d5fa8}
td img{width:120px;border-radius:4px}
"""


def write_html(study: Study, path: Path, title: str = "Capsule Endoscopy Study Report") -> None:
    s = study.summary()
    k = lambda label, val: f'<div class="kpi"><b>{html.escape(str(val))}</b><span>{label}</span></div>'
    kpis = [k("frames", s["frames"]), k("recording", _fmt_t(s["duration_s"])), k("usable", f'{s["usable_pct"]}%'),
            k("frames to review", s["frames_to_review"]), k("review reduction", f'{s["review_reduction_pct"]}%'),
            k("bleeding events", s["events"]["bleeding"]), k("lesion events", s["events"]["lesion"])]
    if "gastric_transit_s" in s:
        kpis += [k("gastric transit (est.)", _fmt_t(s["gastric_transit_s"])),
                 k("small-bowel transit (est.)", _fmt_t(s["small_bowel_transit_s"]))]

    rows = []
    for i, e in enumerate(study.events, 1):
        img = study.annotated.get(e.peak_index)
        thumb = f'<img src="{_b64jpg(img)}" alt="frame {e.peak_index}">' if img is not None else ""
        score = f"{e.peak_score:.1%} of view" if e.kind == "bleeding" else f"{e.peak_score:.2%} area"
        rows.append(f"<tr><td>{i}</td><td><span class='tag {e.kind}'>{e.kind}</span></td>"
                    f"<td>{_fmt_t(e.start_time)} – {_fmt_t(e.end_time)}<br>frames {e.start_index}–{e.end_index}"
                    f" ({e.n_frames} flagged)</td><td>{html.escape(e.region)}</td><td>{score}</td>"
                    f"<td>{thumb}</td></tr>")
    events_html = ("<table><tr><th>#</th><th>Type</th><th>Time</th><th>Region</th><th>Peak</th><th>Peak frame</th>"
                   "</tr>" + "".join(rows) + "</table>") if rows else "<p>No events flagged.</p>"

    landmarks = []
    for name, t in zip(("Pylorus (stomach → small bowel)", "Ileocaecal valve (small bowel → colon)"),
                       study.boundaries):
        near = min(study.overview, key=lambda i: abs(study.results[i].timestamp - t), default=None)
        if near is not None:
            landmarks.append(f'<figure><img src="{_b64jpg(study.overview[near])}" alt="{name}">'
                             f'<figcaption>{name}<br>~{_fmt_t(t)}</figcaption></figure>')
    overview_keys = sorted(study.overview)[:: max(1, len(study.overview) // 24)]
    gallery = "".join(f'<figure><img src="{_b64jpg(study.overview[i])}" alt="frame {i}"><figcaption>'
                      f'#{i} · {_fmt_t(study.results[i].timestamp)} · {study.results[i].region}</figcaption></figure>'
                      for i in overview_keys)

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1>
<div style="color:var(--muted)">Source: {html.escape(s['source'])} · generated {datetime.now():%Y-%m-%d %H:%M}</div>
<div class="warn"><b>Decision-support only.</b> Findings are automated colour/texture heuristics that flag frames for
a qualified reader. They are not a diagnosis, and absence of a flag does not exclude pathology. Region boundaries and
transit times are estimates (fit confidence {s['region_confidence']:.0%}); confirm landmarks visually.</div>
<h2>Summary</h2><div class="kpis">{''.join(kpis)}</div>
<h2>Timeline</h2>{_timeline_svg(study)}
<h2>Flagged events</h2>{events_html}
<h2>Estimated landmarks</h2><div class="grid">{''.join(landmarks) or '<p>Not enough usable frames.</p>'}</div>
<h2>Study overview</h2><div class="grid">{gallery}</div>
</main></body></html>"""
    path.write_text(doc, encoding="utf-8")


def _letterbox(img: np.ndarray, size: int = VIDEO_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.zeros((size, size, 3), np.uint8)
    y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas


def write_highlights(study: Study, path: Path, fps: float = 4.0, per_event: int = 24) -> int:
    """Title card per event followed by its annotated frames. Returns frames written."""
    if not study.events:
        return 0
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (VIDEO_SIZE, VIDEO_SIZE))
    written = 0
    for n, e in enumerate(study.events, 1):
        card = np.zeros((VIDEO_SIZE, VIDEO_SIZE, 3), np.uint8)
        for j, line in enumerate([f"Event {n}/{len(study.events)}: {e.kind.upper()}",
                                  f"{_fmt_t(e.start_time)} - {_fmt_t(e.end_time)}", f"region: {e.region}"]):
            cv2.putText(card, line, (14, 130 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
        for _ in range(int(fps)):
            vw.write(card)
            written += 1
        idx = [i for i in range(e.start_index, e.end_index + 1) if i in study.annotated]
        step = max(1, len(idx) // per_event)
        for i in idx[::step]:
            vw.write(_letterbox(study.annotated[i]))
            written += 1
    vw.release()
    return written


def write_all(study: Study, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"csv": out / "frames.csv", "json": out / "study.json", "html": out / "report.html",
             "video": out / "highlights.mp4"}
    write_frames_csv(study, paths["csv"])
    data = study.to_dict()
    data["frames_flagged"] = [asdict(r) for r in study.results if r.bleeding or r.lesions]
    paths["json"].write_text(json.dumps(data, indent=2))
    write_html(study, paths["html"])
    if not write_highlights(study, paths["video"]):
        paths.pop("video")
    return paths

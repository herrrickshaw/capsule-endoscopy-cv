"""Unit and CLI smoke tests on tiny synthetic arrays (no patient data, no network)."""
from __future__ import annotations

import json
import zlib

import cv2
import numpy as np
import pytest

from capsule_cv import cli
from capsule_cv.analysis import (
    REGION_NAMES,
    FrameAnalyzer,
    FrameResult,
    Study,
    field_of_view_mask,
    group_events,
    region_at,
    segment_regions,
    sharpness_score,
)
from capsule_cv.ingest import HEADER, MAGIC, Frame, encode_message, iter_image_dir, open_source
from capsule_cv.report import _fmt_t, write_frames_csv


def _result(i, t, **kw):
    base = dict(index=i, timestamp=t, brightness=100.0, sharpness=0.9, overexposed=0.0, debris=0.0,
                usable=True, bleeding_score=0.0, bleeding=False)
    base.update(kw)
    return FrameResult(**base)


def test_fmt_t():
    assert _fmt_t(0) == "0:00:00"
    assert _fmt_t(3725) == "1:02:05"


def test_region_at_boundaries():
    assert region_at(5, [10, 20]) == REGION_NAMES[0]
    assert region_at(10, [10, 20]) == REGION_NAMES[1]
    assert region_at(25, [10, 20]) == REGION_NAMES[2]
    assert region_at(5, []) == "unknown"


def test_group_events_gap_and_min_frames():
    rs = [_result(i, i * 1.0, bleeding=i in (0, 1, 2, 10, 20, 21), bleeding_score=0.1 * (i + 1)) for i in range(25)]
    ev = group_events(rs, "bleeding", gap_s=3.0)
    # frames 0-2 and 20-21 qualify; the lone frame 10 is below min_frames
    assert [(e.start_index, e.end_index, e.n_frames) for e in ev] == [(0, 2, 3), (20, 21, 2)]
    assert ev[0].peak_index == 2


def test_group_events_lesion_peak_uses_area():
    small, big = {"kind": "pale", "box": [0, 0, 5, 5], "area": 0.01}, {"kind": "red", "box": [0, 0, 9, 9], "area": 0.05}
    rs = [_result(0, 0.0, lesions=[small]), _result(1, 1.0, lesions=[big])]
    (e,) = group_events(rs, "lesion", gap_s=3.0)
    assert (e.peak_index, e.peak_score) == (1, 0.05)


def test_segment_regions_needs_enough_frames():
    assert segment_regions([_result(i, float(i)) for i in range(10)]) == ([], 0.0)


def test_segment_regions_finds_two_change_points():
    rs = []
    for i in range(300):
        a = 130.0 if i < 60 else 150.0 if i < 220 else 135.0
        rs.append(_result(i, float(i), mean_a=a + (i % 3) * 0.1, mean_b=a / 2))
    bounds, conf = segment_regions(rs)
    assert len(bounds) == 2 and abs(bounds[0] - 60) <= 15 and abs(bounds[1] - 220) <= 15
    assert conf > 0.5


def test_field_of_view_mask_circle_vs_full():
    black = np.zeros((64, 64, 3), np.uint8)
    cv2.circle(black, (32, 32), 28, (120, 120, 200), -1)
    m = field_of_view_mask(black)
    assert m[32, 32] == 255 and m[0, 0] == 0
    assert field_of_view_mask(np.full((64, 64, 3), 150, np.uint8)).all()


def test_sharpness_blur_scores_lower():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4)
    region = np.ones((128, 128), bool)
    assert sharpness_score(blurred, region) < sharpness_score(sharp, region)


def test_analyzer_flags_saturated_red_as_bleeding():
    img = np.full((128, 128, 3), (110, 120, 200), np.uint8)  # pink-ish mucosa (BGR)
    img[30:90, 30:90] = (10, 10, 200)  # saturated red patch, R/G >> 2.2
    res, red_mask = FrameAnalyzer().analyze(Frame(0, 0.0, img))
    assert res.bleeding and res.bleeding_score > 0.1
    assert red_mask.dtype == np.uint8 and red_mask.max() == 255


def test_encode_message_header_roundtrip():
    msg = encode_message(7, 1.5, np.full((16, 16, 3), 90, np.uint8))
    magic, seq, ts, length, crc = HEADER.unpack(msg[:HEADER.size])
    payload = msg[HEADER.size:]
    assert (magic, seq, ts, length) == (MAGIC, 7, 1.5, len(payload))
    assert crc == zlib.crc32(payload)


def test_iter_image_dir_skips_unreadable(tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "b.png").write_bytes(b"not an image")
    cv2.imwrite(str(tmp_path / "c.png"), np.full((8, 8, 3), 5, np.uint8))
    frames = list(open_source(tmp_path, fps=2.0))
    assert [f.timestamp for f in iter_image_dir(tmp_path, 2.0)] == [0.0, 1.0]  # index 1 (b.png) is skipped
    assert len(frames) == 2


def test_open_source_missing_video_raises(tmp_path):
    with pytest.raises(IOError):
        list(open_source(tmp_path / "nope.mp4"))


def test_empty_study_summary_and_csv(tmp_path):
    st = Study("empty", [], [], [], 0.0, {}, {})
    s = st.summary()
    assert s["frames"] == 0 and s["review_reduction_pct"] == 0
    write_frames_csv(st, tmp_path / "f.csv")
    assert (tmp_path / "f.csv").read_text().count("\n") == 1


def test_cli_synth_then_analyze(tmp_path, capsys):
    video = tmp_path / "s" / "c.mp4"
    assert cli.main(["synth", "--out", str(video), "--frames", "60", "--size", "96"]) == 0
    assert video.exists()
    out = tmp_path / "rep"
    assert cli.main(["analyze", str(video), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert '"frames": 60' in printed
    data = json.loads((out / "study.json").read_text())
    assert data["summary"]["frames"] == 60
    assert (out / "report.html").exists() and (out / "frames.csv").exists()


def test_cli_send_and_receive_use_injected_transport(monkeypatch, tmp_path, capsys):
    sent = {}
    monkeypatch.setattr(cli, "send_frames", lambda frames, host, port, rate: sent.update(h=host, p=port, r=rate) or 3)
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((8, 8, 3), np.uint8))
    assert cli.main(["send", str(tmp_path), "--port", "6000", "--rate", "2"]) == 0
    assert sent == {"h": "127.0.0.1", "p": 6000, "r": 2.0} and "sent 3 frames" in capsys.readouterr().out

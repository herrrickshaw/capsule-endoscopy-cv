"""Threshold, bookkeeping and I/O-edge tests for capsule_cv (synthetic arrays only; loopback sockets at most).

Nothing here asserts a clinical fact: it pins the code's own arithmetic and thresholds.
"""
from __future__ import annotations

import math
import runpy
import socket
import struct
import threading
import zlib
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from capsule_cv import analysis, cli, ingest, report, synth
from capsule_cv.analysis import (
    Event,
    FrameAnalyzer,
    FrameResult,
    Study,
    Thresholds,
    _seg_cost,
    analyze_stream,
    annotate,
    group_events,
    region_at,
    segment_regions,
)
from capsule_cv.ingest import HEADER, MAGIC, Frame, Receiver, encode_message


def res(i, t=None, **kw):
    base = dict(index=i, timestamp=float(i if t is None else t), brightness=100.0, sharpness=0.9,
                overexposed=0.0, debris=0.0, usable=True, bleeding_score=0.0, bleeding=False)
    base.update(kw)
    return FrameResult(**base)


def uniform(bgr, size=64):
    return np.full((size, size, 3), bgr, np.uint8)


class TestThresholds:
    def test_defaults(self):
        t = Thresholds()
        assert (t.min_brightness, t.max_brightness, t.min_sharpness) == (35.0, 225.0, 0.44)
        assert (t.max_overexposed, t.max_debris, t.bleed_fraction, t.bleed_red_ratio) == (0.20, 0.45, 0.03, 2.2)
        assert (t.lesion_min_area, t.lesion_max_area) == (0.0015, 0.20)
        assert (t.pale_delta_l, t.red_delta_a) == (28.0, 20.0)
        assert (t.duplicate_hist_corr, t.duplicate_mad, t.event_gap_s) == (0.98, 5.0, 3.0)

    def test_analyzer_uses_given_thresholds(self):
        th = Thresholds(min_brightness=1.0)
        assert FrameAnalyzer(th).th is th and FrameAnalyzer().th == Thresholds()


class TestQualityVerdict:
    # min_sharpness=0 isolates the other gates (a uniform frame has no edges -> sharpness 0)
    def _usable(self, bgr, **th):
        r, _ = FrameAnalyzer(Thresholds(min_sharpness=0.0, **th)).analyze(Frame(0, 0.0, uniform(bgr)))
        return r

    def test_brightness_bounds_are_inclusive(self):
        assert self._usable((100, 100, 100), min_brightness=100.0).usable
        assert not self._usable((100, 100, 100), min_brightness=100.5).usable
        assert self._usable((100, 100, 100), max_brightness=100.0).usable
        assert not self._usable((100, 100, 100), max_brightness=99.5).usable

    def test_overexposure_gate(self):
        r = self._usable((255, 255, 255), max_brightness=255.0)
        assert r.overexposed == 1.0 and not r.usable
        assert self._usable((255, 255, 255), max_brightness=255.0, max_overexposed=1.0).usable

    def test_debris_gate(self):
        r = self._usable((0, 200, 200))                      # yellow-green hue, saturated
        assert r.debris == 1.0 and not r.usable
        assert self._usable((0, 200, 200), max_debris=1.0).usable

    def test_sharpness_gate_uses_min_sharpness(self):
        rng = np.random.default_rng(0)
        noisy = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
        r, _ = FrameAnalyzer(Thresholds(min_sharpness=0.0)).analyze(Frame(0, 0.0, noisy))
        assert r.sharpness > 0.0
        strict, _ = FrameAnalyzer(Thresholds(min_sharpness=r.sharpness + 0.01)).analyze(Frame(0, 0.0, noisy))
        assert strict.usable is False and r.usable is True

    def test_result_fields_are_rounded(self):
        r = self._usable((100, 100, 100))
        assert r.brightness == 100.0 and r.index == 0 and r.timestamp == 0.0 and r.region == ""
        assert r.sharpness == round(r.sharpness, 3)


class TestAnalyzerVerdictWiring:
    def _analyzer(self, monkeypatch, usable, score=0.0, lesions=None):
        fa = FrameAnalyzer()
        monkeypatch.setattr(fa, "_quality", lambda *a: {
            "brightness": 100.0, "sharpness": 0.9, "specular": np.zeros((64, 64), bool), "overexposed": 0.0,
            "debris_mask": np.zeros((64, 64), bool), "debris": 0.0, "usable": usable})
        monkeypatch.setattr(fa, "_blood", lambda *a: (np.zeros((64, 64), np.uint8), score))
        monkeypatch.setattr(fa, "_lesions", lambda *a: (list(lesions or []), 1.0, 2.0))
        return fa

    def test_bleeding_threshold_is_inclusive(self, monkeypatch):
        f = Frame(0, 0.0, uniform((100, 100, 100)))
        assert self._analyzer(monkeypatch, True, 0.03).analyze(f)[0].bleeding is True
        assert self._analyzer(monkeypatch, True, 0.0299).analyze(f)[0].bleeding is False
        r = self._analyzer(monkeypatch, True, 0.123456).analyze(f)[0]
        assert r.bleeding_score == 0.1235

    def test_lesions_dropped_on_unusable_frames_but_stats_kept(self, monkeypatch):
        les = [{"kind": "pale", "box": [1, 2, 3, 4], "area": 0.01}]
        f = Frame(0, 0.0, uniform((100, 100, 100)))
        r = self._analyzer(monkeypatch, False, lesions=les).analyze(f)[0]
        assert r.lesions == [] and (r.mean_a, r.mean_b) == (1.0, 2.0)
        assert self._analyzer(monkeypatch, True, lesions=les).analyze(f)[0].lesions == les


class TestDuplicates:
    def test_mad_boundary_is_inclusive(self):
        img = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        fa = FrameAnalyzer(Thresholds(duplicate_mad=0.0, duplicate_hist_corr=-1.0))
        assert [fa.analyze(Frame(i, i, img))[0].duplicate for i in range(2)] == [False, True]
        fa = FrameAnalyzer(Thresholds(duplicate_mad=0.0, duplicate_hist_corr=-1.0))
        changed = img.copy()
        changed[:8] = 255 - changed[:8]
        assert [fa.analyze(Frame(i, i, im))[0].duplicate for i, im in enumerate((img, changed))] == [False, False]

    def test_hist_corr_gate(self):
        img = np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        fa = FrameAnalyzer(Thresholds(duplicate_hist_corr=1.01))
        assert [fa.analyze(Frame(i, i, img))[0].duplicate for i in range(2)] == [False, False]

    def test_reference_stays_at_last_retained_frame(self):
        a = np.random.default_rng(3).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        b = np.random.default_rng(4).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        fa = FrameAnalyzer()
        flags = [fa.analyze(Frame(i, i, im))[0].duplicate for i, im in enumerate((a, a, b, b, a))]
        assert flags == [False, True, False, True, False]


class TestGroupEvents:
    def test_gap_boundary_is_strict(self):
        rs = [res(0, 0.0, bleeding=True), res(1, 3.0, bleeding=True), res(2, 6.01, bleeding=True),
              res(3, 7.0, bleeding=True)]
        ev = group_events(rs, "bleeding", gap_s=3.0)
        # 0.0 -> 3.0 is exactly the gap (same run); 3.0 -> 6.01 exceeds it (new run)
        assert [(e.start_index, e.end_index) for e in ev] == [(0, 1), (2, 3)]

    def test_min_frames_and_event_fields(self):
        rs = [res(0, 0.0, bleeding=True, bleeding_score=0.1, region="colon"),
              res(1, 1.0, bleeding=True, bleeding_score=0.25, region="stomach"),
              res(5, 9.0, bleeding=True, bleeding_score=0.3)]
        assert len(group_events(rs, "bleeding", 3.0)) == 1
        both = group_events(rs, "bleeding", 3.0, min_frames=1)
        assert len(both) == 2
        e = both[0]
        assert (e.kind, e.start_time, e.end_time, e.peak_index, e.peak_score, e.n_frames) == (
            "bleeding", 0.0, 1.0, 1, 0.25, 2)
        assert e.region == "stomach"          # region of the peak frame

    def test_no_flags_no_events(self):
        assert group_events([res(i) for i in range(5)], "bleeding", 3.0) == []
        assert group_events([res(i) for i in range(5)], "lesion", 3.0) == []
        assert group_events([], "lesion", 3.0) == []

    def test_lesion_kind_ignores_bleeding_flag(self):
        les = [{"kind": "red", "box": [0, 0, 4, 4], "area": 0.02}]
        rs = [res(0, 0.0, bleeding=True), res(1, 1.0, lesions=les), res(2, 2.0, lesions=les)]
        (e,) = group_events(rs, "lesion", 3.0)
        assert (e.start_index, e.end_index, e.n_frames) == (1, 2, 2)

    def test_region_at_exact_boundaries(self):
        assert region_at(9.99, [10.0, 20.0]) == "stomach" and region_at(10.0, [10.0, 20.0]) == "small bowel"
        assert region_at(19.99, [10.0, 20.0]) == "small bowel" and region_at(20.0, [10.0, 20.0]) == "colon"
        assert region_at(1.0, [5.0]) == "unknown"


class TestSummary:
    def _study(self, results, bounds=()):
        return Study("src", results, [], list(bounds), 0.756, {}, {})

    def test_exact_counts(self):
        rs = [
            res(0, 0.0),                                                    # usable, kept
            res(1, 1.0, duplicate=True),                                    # usable duplicate (not read)
            res(2, 2.0, usable=False, bleeding=True),                       # unusable but flagged -> read
            res(3, 3.0, usable=False),                                      # unusable, nothing flagged
            res(4, 10.0, lesions=[{"kind": "pale", "box": [0, 0, 1, 1], "area": 0.1}]),   # usable, kept
        ]
        s = self._study(rs).summary()
        assert s["frames"] == 5 and s["duration_s"] == 10.0 and s["usable_frames"] == 3
        assert s["usable_pct"] == 60.0 and s["duplicate_frames"] == 1
        assert s["frames_to_review"] == 3 and s["review_reduction_pct"] == 40.0
        assert (s["bleeding_frames"], s["lesion_frames"]) == (1, 1)
        assert s["events"] == {"bleeding": 0, "lesion": 0} and s["region_confidence"] == 0.76
        assert "gastric_transit_s" not in s and s["source"] == "src"

    def test_transit_times_need_two_boundaries(self):
        rs = [res(0, 5.0), res(1, 100.0)]
        s = self._study(rs, [25.0, 65.0]).summary()
        assert s["gastric_transit_s"] == 20.0 and s["small_bowel_transit_s"] == 40.0
        assert "gastric_transit_s" not in self._study(rs, [25.0]).summary()

    def test_single_frame_has_zero_duration_and_to_dict(self):
        st = Study("s", [res(0, 7.0)], [Event("bleeding", 0, 0, 7.0, 7.0, 0, 0.1, 1)], [1.0, 2.0], 0.5, {}, {})
        assert st.summary()["duration_s"] == 0.0
        d = st.to_dict()
        assert set(d) == {"summary", "boundaries", "events"} and d["events"][0]["kind"] == "bleeding"
        assert d["boundaries"] == [1.0, 2.0]


class TestSegmentation:
    def test_seg_cost_matches_within_segment_variance(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 3))
        cs = np.vstack([np.zeros(3), np.cumsum(X, 0)])
        cs2 = np.vstack([np.zeros(3), np.cumsum(X * X, 0)])
        expected = ((X[2:9] - X[2:9].mean(0)) ** 2).sum()
        assert _seg_cost(cs, cs2, 2, 9) == pytest.approx(expected)
        a, b = np.array([0, 4]), np.array([6, 12])
        got = _seg_cost(cs, cs2, a, b)
        assert got == pytest.approx([((X[0:6] - X[0:6].mean(0)) ** 2).sum(), ((X[4:12] - X[4:12].mean(0)) ** 2).sum()])

    def test_empty_segment_costs_zero(self):
        cs = np.zeros((4, 2))
        assert _seg_cost(cs, cs, 2, 2) == 0.0

    def test_too_few_bins_returns_nothing(self):
        rs = [res(i, mean_a=float(i % 7)) for i in range(60)]
        assert segment_regions(rs, max_bins=5) == ([], 0.0)

    def test_unusable_and_duplicate_frames_are_ignored(self):
        rs = [res(i, mean_a=float(i % 5), usable=False) for i in range(100)]
        assert segment_regions(rs) == ([], 0.0)
        rs = [res(i, mean_a=float(i % 5), duplicate=True) for i in range(100)]
        assert segment_regions(rs) == ([], 0.0)
        assert segment_regions([res(i, mean_a=float(i)) for i in range(29)]) == ([], 0.0)

    def test_confidence_in_unit_interval_and_boundaries_ordered(self):
        rs = [res(i, mean_a=(10.0 if i < 50 else 40.0 if i < 150 else 20.0) + (i % 2) * 0.1, mean_b=1.0)
              for i in range(200)]
        b, conf = segment_regions(rs, max_bins=40)
        assert 0.0 < conf <= 1.0 and b[0] < b[1] and abs(b[0] - 50) <= 12 and abs(b[1] - 150) <= 12

    def test_constant_signal_has_zero_confidence_or_no_split(self):
        rs = [res(i, mean_a=5.0, mean_b=5.0, sharpness=0.5) for i in range(100)]
        b, conf = segment_regions(rs)
        assert conf == 0.0


class TestAnnotateAndThumb:
    def test_thumb_downscales_but_never_upscales(self):
        big = uniform((10, 20, 30), 640)
        assert analysis._thumb(big).shape[:2] == (320, 320)
        wide = np.zeros((200, 800, 3), np.uint8)
        assert analysis._thumb(wide).shape[:2] == (80, 320)
        small = uniform((1, 2, 3), 100)
        t = analysis._thumb(small)
        assert t.shape == small.shape and t is not small

    def test_annotation_draws_label_boxes_and_tint(self):
        img = uniform((90, 120, 160), 100)
        mask = np.zeros((100, 100), np.uint8)
        mask[40:60, 40:60] = 255
        r = res(12, 3.5, bleeding=True, bleeding_score=0.5,
                lesions=[{"kind": "pale", "box": [10, 10, 8, 8], "area": 0.01},
                         {"kind": "red", "box": [70, 70, 8, 8], "area": 0.01}])
        out = annotate(img, r, mask)
        assert out.shape == img.shape and img[50, 50].tolist() == [90, 120, 160]
        assert out[50, 50].tolist() != [90, 120, 160]              # tinted blood area
        assert out[20, 80].tolist() == [90, 120, 160]              # outside mask unchanged
        assert (out[6:6 + 13, 6:6 + 120] != img[6:6 + 13, 6:6 + 120]).any()   # label text drawn

    def test_no_tint_without_bleeding(self):
        img = uniform((90, 120, 160), 100)
        mask = np.full((100, 100), 255, np.uint8)
        out = annotate(img, res(0, bleeding=False), mask)
        assert out[50, 50].tolist() == [90, 120, 160]
        assert annotate(img, res(0, bleeding=True), None)[50, 50].tolist() == [90, 120, 160]


class TestAnalyzeStream:
    @pytest.fixture
    def fake_analyzer(self, monkeypatch):
        class FA:
            def __init__(self, thresholds=None):
                self.th = thresholds or Thresholds()

            def analyze(self, f):
                i = f.index
                return res(i, f.timestamp, bleeding=(i % 2 == 0), bleeding_score=0.1,
                           usable=(i % 5 != 0)), np.zeros((16, 16), np.uint8)

        monkeypatch.setattr(analysis, "FrameAnalyzer", FA)

    def _frames(self, n):
        return (Frame(i, i * 0.5, uniform((i, i, i), 16)) for i in range(n))

    def test_annotation_cap_and_overview_decimation(self, monkeypatch, fake_analyzer):
        monkeypatch.setattr(analysis, "MAX_ANNOTATED", 4)
        monkeypatch.setattr(analysis, "OVERVIEW_TARGET", 3)
        st = analyze_stream(self._frames(120), "x")
        assert len(st.results) == 120 and len(st.annotated) == 4
        assert sorted(st.annotated) == [0, 2, 4, 6]
        assert len(st.overview) <= 2 * 3
        assert all(k % 5 != 0 for k in st.overview)                    # only usable frames
        assert math.gcd(*st.overview) >= 2            # kept frames share the decimation stride

    def test_regions_assigned_from_boundaries_and_events_sorted(self, monkeypatch, fake_analyzer):
        monkeypatch.setattr(analysis, "segment_regions", lambda rs: ([5.0, 12.0], 0.9))
        st = analyze_stream(self._frames(40), "src")
        assert st.boundaries == [5.0, 12.0] and st.region_confidence == 0.9 and st.source == "src"
        assert [r.region for r in st.results[:3]] == ["stomach", "stomach", "stomach"]
        by_t = {r.timestamp: r.region for r in st.results}
        assert by_t[5.0] == "small bowel" and by_t[11.5] == "small bowel" and by_t[12.0] == "colon"
        starts = [e.start_time for e in st.events]
        assert starts == sorted(starts)

    def test_event_gap_comes_from_thresholds(self, monkeypatch, fake_analyzer):
        # bleeding on even frames, 1.0 s apart: gap_s=0.5 splits every pair into singletons -> no events
        st = analyze_stream(self._frames(20), thresholds=Thresholds(event_gap_s=0.5))
        assert [e for e in st.events if e.kind == "bleeding"] == []
        st = analyze_stream(self._frames(20), thresholds=Thresholds(event_gap_s=1.0))
        assert len([e for e in st.events if e.kind == "bleeding"]) == 1

    def test_progress_lines(self, fake_analyzer, capsys):
        analyze_stream(self._frames(12), progress_every=5)
        assert capsys.readouterr().out.splitlines() == ["  analysed 5 frames", "  analysed 10 frames"]
        analyze_stream(self._frames(12), progress_every=0)
        assert capsys.readouterr().out == ""


class TestIngest:
    def test_video_rate_and_fallbacks(self, tmp_path):
        p = tmp_path / "v.mp4"
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter.fourcc(*"mp4v"), 5.0, (32, 32))
        for i in range(4):
            vw.write(uniform((i * 40, 60, 90), 32))
        vw.release()
        native = list(ingest.iter_video(p))
        assert len(native) == 4 and [f.index for f in native] == [0, 1, 2, 3]
        assert [f.timestamp for f in native] == pytest.approx([0.0, 0.2, 0.4, 0.6], abs=1e-6)
        over = list(ingest.iter_video(p, fps=2.0))
        assert [f.timestamp for f in over] == [0.0, 0.5, 1.0, 1.5]

    def test_zero_fps_container_falls_back_to_two(self, monkeypatch):
        class Cap:
            def __init__(self, path):
                self.n = 0

            def isOpened(self):
                return True

            def get(self, prop):
                return 0.0

            def read(self):
                self.n += 1
                return (self.n <= 3), uniform((1, 1, 1), 4)

            def release(self):
                Cap.released = True

        monkeypatch.setattr(ingest.cv2, "VideoCapture", Cap)
        frames = list(ingest.iter_video("x.mp4"))
        assert [f.timestamp for f in frames] == [0.0, 0.5, 1.0] and Cap.released is True

    def test_unopenable_video_names_the_path(self, tmp_path):
        with pytest.raises(IOError, match="cannot open video: .*nope.mp4"):
            list(ingest.iter_video(tmp_path / "nope.mp4"))

    def test_image_dir_sorted_filtered_and_rate(self, tmp_path):
        for name in ("b.PNG", "a.jpg", "c.txt", "d.bmp"):
            if name.endswith("txt"):
                (tmp_path / name).write_text("x")
            else:
                cv2.imwrite(str(tmp_path / name.lower()), uniform((len(name), 5, 9), 8))
        frames = list(ingest.open_source(tmp_path))
        assert [f.timestamp for f in frames] == [0.0, 0.5, 1.0]
        assert [f.timestamp for f in ingest.open_source(tmp_path, fps=4.0)] == [0.0, 0.25, 0.5]
        assert ingest.IMAGE_EXTS == {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def test_open_source_file_passes_fps(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(ingest, "iter_video", lambda p, fps: seen.update(p=p, fps=fps) or iter(()))
        list(ingest.open_source(tmp_path / "x.mp4", fps=3.0))
        assert seen["fps"] == 3.0

    def test_encode_failure_raises(self, monkeypatch):
        monkeypatch.setattr(ingest.cv2, "imencode", lambda *a, **k: (False, None))
        with pytest.raises(ValueError, match="JPEG encode failed"):
            encode_message(0, 0.0, uniform((1, 1, 1), 4))

    def test_message_layout(self):
        assert HEADER.size == struct.calcsize("!4sIdII") == 24 and MAGIC == b"CAPS"
        msg = encode_message(3, 2.5, uniform((9, 9, 9), 8), quality=50)
        assert msg[:4] == b"CAPS"
        assert len(msg) == HEADER.size + HEADER.unpack(msg[:HEADER.size])[3]

    def test_send_frames_counts_and_throttles(self, monkeypatch):
        sent, sleeps = [], []

        class Sock:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def sendall(self, b):
                sent.append(b)

        monkeypatch.setattr(ingest.socket, "create_connection", lambda addr: Sock())
        monkeypatch.setattr(ingest.time, "sleep", lambda s: sleeps.append(s))
        frames = [Frame(i, i * 0.5, uniform((5, 5, 5), 8)) for i in range(3)]
        assert ingest.send_frames(frames, "h", 1, rate=4.0) == 3
        assert len(sent) == 3 and sleeps == [0.25] * 3
        sleeps.clear()
        assert ingest.send_frames(frames, "h", 1) == 3 and sleeps == []


def _serve(rx, payloads):
    def run():
        with socket.create_connection(("127.0.0.1", rx.port)) as s:
            for p in payloads:
                s.sendall(p)
    t = threading.Thread(target=run)
    t.start()
    return t


class TestReceiver:
    IMG = np.full((32, 32, 3), 120, np.uint8)

    def test_port_and_archive_dir(self, tmp_path):
        rx = Receiver("127.0.0.1", 0, archive_dir=tmp_path / "a" / "b")
        assert rx.port > 0 and (tmp_path / "a" / "b").is_dir()
        assert rx.stats.received == rx.stats.lost == rx.stats.corrupted == rx.stats.bytes == 0
        rx._server.close()

    def test_bad_magic_raises(self):
        rx = Receiver("127.0.0.1", 0, accept_timeout=10)
        t = _serve(rx, [HEADER.pack(b"XXXX", 0, 0.0, 0, 0)])
        with pytest.raises(ValueError, match="stream desynchronised"):
            list(rx.frames())
        t.join()

    def test_truncated_payload_ends_stream_quietly(self):
        rx = Receiver("127.0.0.1", 0, accept_timeout=10)
        msg = encode_message(0, 0.0, self.IMG)
        t = _serve(rx, [encode_message(0, 0.0, self.IMG), msg[: HEADER.size + 3]])
        frames = list(rx.frames())
        t.join()
        assert len(frames) == 1 and rx.stats.received == 1

    def test_undecodable_payload_counts_as_corrupted_and_bytes_accumulate(self):
        rx = Receiver("127.0.0.1", 0, accept_timeout=10)
        junk = b"not a jpeg at all"
        bad = HEADER.pack(MAGIC, 0, 0.0, len(junk), zlib.crc32(junk)) + junk
        good = encode_message(1, 1.0, self.IMG)
        t = _serve(rx, [bad, good])
        frames = list(rx.frames())
        t.join()
        assert len(frames) == 1 and frames[0].index == 0       # output index counts delivered frames only
        assert (rx.stats.received, rx.stats.corrupted, rx.stats.lost) == (1, 1, 0)
        assert rx.stats.bytes == len(bad) + len(good)

    def test_out_of_order_sequence_is_not_loss(self):
        rx = Receiver("127.0.0.1", 0, accept_timeout=10)
        t = _serve(rx, [encode_message(5, 0.0, self.IMG), encode_message(2, 1.0, self.IMG),
                        encode_message(3, 2.0, self.IMG)])
        list(rx.frames())
        t.join()
        assert rx.stats.lost == 0 and rx.stats.gaps == []

    def test_gap_accounting_and_archive_names(self, tmp_path):
        rx = Receiver("127.0.0.1", 0, archive_dir=tmp_path, accept_timeout=10)
        t = _serve(rx, [encode_message(0, 0.0, self.IMG), encode_message(4, 1.0, self.IMG),
                        encode_message(9, 2.0, self.IMG)])
        list(rx.frames())
        t.join()
        assert rx.stats.gaps == [(1, 3), (5, 8)] and rx.stats.lost == 7
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "frame_0000000.jpg", "frame_0000004.jpg", "frame_0000009.jpg"]

    def test_accept_timeout(self):
        rx = Receiver("127.0.0.1", 0, accept_timeout=0.05)
        with pytest.raises(socket.timeout):
            next(rx.frames())
        rx._server.close()


class TestReport:
    def test_fmt_t_rounding(self):
        assert report._fmt_t(59.4) == "0:00:59" and report._fmt_t(59.6) == "0:01:00"
        assert report._fmt_t(36000) == "10:00:00"

    def test_timeline_needs_two_frames(self):
        assert report._timeline_svg(Study("s", [res(0)], [], [], 0.0, {}, {})) == ""

    def test_timeline_regions_and_events(self):
        rs = [res(i, float(i), bleeding_score=0.2 if i == 5 else 0.0, usable=i != 3) for i in range(10)]
        ev = [Event("lesion", 4, 6, 4.0, 6.0, 5, 0.01, 3)]
        svg = report._timeline_svg(Study("s", rs, ev, [3.0, 7.0], 0.5, {}, {}))
        for name in ("stomach", "small bowel", "colon"):
            assert f">{name}</text>" in svg
        assert svg.count("<line ") == 1 and "#1d5fa8" in svg and "<polyline" in svg
        no_bounds = report._timeline_svg(Study("s", rs, [], [], 0.0, {}, {}))
        assert ">unknown</text>" in no_bounds and "stomach" not in no_bounds

    def test_no_events_means_no_video_and_zero_frames(self, tmp_path):
        st = Study("s", [res(0), res(1)], [], [], 0.0, {}, {})
        assert report.write_highlights(st, tmp_path / "h.mp4") == 0
        paths = report.write_all(st, tmp_path / "out")
        assert set(paths) == {"csv", "json", "html"} and not (tmp_path / "out" / "highlights.mp4").exists()

    def test_highlights_cards_and_frame_sampling(self, tmp_path):
        ev = [Event("bleeding", 0, 9, 0.0, 9.0, 3, 0.2, 10, "colon")]
        annotated = {i: uniform((i * 10, 50, 50), 100) for i in range(10)}
        st = Study("s", [res(i) for i in range(10)], ev, [], 0.0, annotated, {})
        n = report.write_highlights(st, tmp_path / "h.mp4", fps=4.0, per_event=3)
        # 4 title-card frames + every 3rd of 10 annotated frames (indices 0,3,6,9)
        assert n == 4 + 4 and (tmp_path / "h.mp4").exists()

    def test_letterbox_centres_and_scales(self):
        wide = np.full((50, 100, 3), 200, np.uint8)
        out = report._letterbox(wide, 200)
        assert out.shape == (200, 200, 3) and out[100, 100].tolist() == [200, 200, 200]
        assert out[0, 100].tolist() == [0, 0, 0] and out[199, 100].tolist() == [0, 0, 0]

    def test_csv_columns_and_lesion_kinds(self, tmp_path):
        les = [{"kind": "red", "box": [0, 0, 1, 1], "area": 0.1}, {"kind": "pale", "box": [0, 0, 1, 1], "area": 0.1},
               {"kind": "red", "box": [0, 0, 1, 1], "area": 0.1}]
        st = Study("s", [res(2, 1.5, lesions=les, region="colon", usable=False, duplicate=True)], [], [], 0, {}, {})
        report.write_frames_csv(st, tmp_path / "f.csv")
        header, row = (tmp_path / "f.csv").read_text().splitlines()
        assert header.split(",")[:3] == ["index", "timestamp", "region"]
        cells = dict(zip(header.split(","), row.split(",")))
        assert cells["timestamp"] == "1.500" and cells["usable"] == "0" and cells["duplicate"] == "1"
        assert cells["n_lesions"] == "3" and cells["lesion_kinds"] == "pale|red" and cells["region"] == "colon"

    def test_html_escapes_title_and_source_and_handles_empty(self, tmp_path):
        st = Study("<script>x</script>", [res(0), res(1)], [], [], 0.5, {}, {})
        p = tmp_path / "r.html"
        report.write_html(st, p, title="<b>T</b>")
        text = p.read_text()
        assert "<script>x</script>" not in text and "&lt;script&gt;" in text and "&lt;b&gt;T&lt;/b&gt;" in text
        assert "No events flagged." in text and "Not enough usable frames." in text
        assert "Decision-support only" in text

    def test_html_event_rows_and_transit_kpis(self, tmp_path):
        rs = [res(i, float(i * 10)) for i in range(6)]
        ev = [Event("bleeding", 1, 2, 10.0, 20.0, 1, 0.1234, 2, "stomach"),
              Event("lesion", 3, 4, 30.0, 40.0, 3, 0.0123, 2, "colon")]
        img = uniform((10, 20, 30), 32)
        st = Study("s", rs, ev, [15.0, 35.0], 0.8, {1: img, 3: img}, {0: img, 2: img})
        p = tmp_path / "r.html"
        report.write_html(st, p)
        text = p.read_text()
        assert "12.3% of view" in text and "1.23% area" in text
        assert "gastric transit (est.)" in text and "Pylorus" in text and "Ileocaecal valve" in text
        assert text.count("<tr><td>") == 2

    def test_b64_helper(self):
        assert report._b64jpg(uniform((1, 2, 3), 8)).startswith("data:image/jpeg;base64,")


class TestSynth:
    def test_region_boundaries_follow_fractions(self):
        cap = synth.SyntheticCapsule(n_frames=200, size=64, seed=1)
        assert cap.boundaries == [30, 150]
        assert [cap.region(i) for i in (0, 29, 30, 149, 150, 199)] == [
            "stomach", "stomach", "small bowel", "small bowel", "colon", "colon"]

    def test_span_membership_is_half_open(self):
        cap = synth.SyntheticCapsule(n_frames=200, size=64, seed=1)
        a, b = cap.truth["bleeding"][0]
        assert (b - a) == 18 and cap._in("bleeding", a) and cap._in("bleeding", b - 1) and not cap._in("bleeding", b)

    def test_ground_truth_payload(self):
        gt = synth.SyntheticCapsule(n_frames=100, size=64, fps=5.0).ground_truth()
        assert gt["n_frames"] == 100 and gt["fps"] == 5.0 and gt["size"] == 64
        assert gt["region_boundaries_s"] == [3.0, 15.0] and gt["region_boundaries_frame"] == [15, 75]
        assert set(gt["segments_frame"]) == {"bleeding", "lesion_pale", "lesion_red", "blur", "bubbles",
                                             "debris", "dwell"}

    def test_frames_are_deterministic_and_timestamped(self):
        a = list(synth.SyntheticCapsule(n_frames=6, size=64, fps=2.0, seed=4).frames())
        b = list(synth.SyntheticCapsule(n_frames=6, size=64, fps=2.0, seed=4).frames())
        assert [f.timestamp for f in a] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        assert all(np.array_equal(x.image, y.image) for x, y in zip(a, b)) and a[0].image.shape == (64, 64, 3)
        assert a[0].image[0, 0].tolist() == [0, 0, 0]          # outside the circular field of view

    def test_write_video_as_images(self, tmp_path):
        out = synth.write_video(tmp_path / "pngs", n_frames=5, size=64, as_images=True)
        assert sorted(p.name for p in out.glob("*.png")) == [f"frame_{i:06d}.png" for i in range(5)]
        assert (out / "truth.json").exists()

    def test_write_video_mp4_truth_sidecar(self, tmp_path):
        out = synth.write_video(tmp_path / "x" / "c.mp4", n_frames=8, size=64)
        assert out.exists() and out.with_suffix(".truth.json").exists()


class TestCli:
    def test_receive_subcommand(self, monkeypatch, tmp_path, capsys):
        class FakeRx:
            port = 4242
            stats = SimpleNamespace(received=7, lost=2, corrupted=1, bytes=999)

            def __init__(self, host, port, archive_dir=None):
                FakeRx.args = (host, port, archive_dir)

            def frames(self):
                return iter(())

        monkeypatch.setattr(cli, "Receiver", FakeRx)
        assert cli.main(["receive", "--host", "127.0.0.1", "--port", "0", "--archive", str(tmp_path / "a"),
                         "--out", str(tmp_path / "rep")]) == 0
        out = capsys.readouterr().out
        assert FakeRx.args == ("127.0.0.1", 0, str(tmp_path / "a"))
        assert "listening on 127.0.0.1:4242 ..." in out
        assert "received=7 lost=2 corrupted=1 bytes=999" in out
        assert (tmp_path / "rep" / "study.json").exists()

    def test_synth_images_flag(self, tmp_path, capsys):
        out = tmp_path / "frames"
        assert cli.main(["synth", "--out", str(out), "--frames", "4", "--size", "64", "--images",
                         "--fps", "2", "--seed", "1"]) == 0
        assert str(out) in capsys.readouterr().out and (out / "truth.json").exists()

    def test_subcommand_required_and_defaults(self):
        with pytest.raises(SystemExit):
            cli.main([])

    def test_module_entry_point(self, monkeypatch):
        monkeypatch.setattr(cli, "main", lambda *a: 3)
        with pytest.raises(SystemExit) as e:
            runpy.run_module("capsule_cv", run_name="__main__")
        assert e.value.code == 3


class TestBoundaryPins:
    def test_duplicate_correlation_threshold_is_inclusive(self, monkeypatch):
        monkeypatch.setattr(analysis.cv2, "compareHist", lambda *a, **k: 0.98)
        img = uniform((100, 100, 100))
        fa = FrameAnalyzer()
        assert [fa.analyze(Frame(i, i, img))[0].duplicate for i in range(2)] == [False, True]

    def test_reference_does_not_drift_through_duplicates(self):
        # uniform frames 100 -> 104 -> 108: each step is within the MAD limit (5) of its neighbour,
        # but 108 is 8 away from the last *retained* frame (100), so it must not be a duplicate
        fa = FrameAnalyzer()
        flags = [fa.analyze(Frame(i, i, uniform((v, v, v))))[0].duplicate for i, v in enumerate((100, 104, 108))]
        assert flags == [False, True, False]

    @pytest.mark.parametrize("corner,full", [(20, True), (19, False)])
    def test_field_of_view_corner_threshold(self, corner, full):
        img = np.full((64, 64, 3), 150, np.uint8)
        c = 4
        for ys, xs in ((slice(0, c), slice(0, c)), (slice(0, c), slice(-c, None)),
                       (slice(-c, None), slice(0, c)), (slice(-c, None), slice(-c, None))):
            img[ys, xs] = corner
        assert bool(analysis.field_of_view_mask(img).all()) is full

    def test_flat_frame_has_zero_sharpness(self):
        flat = np.full((64, 64), 120, np.uint8)
        assert analysis.sharpness_score(flat, np.ones((64, 64), bool)) == 0.0

    def test_segment_regions_minimum_is_thirty_usable_frames(self):
        def frames(n):
            return [res(i, float(i), mean_a=(10.0 if i < n // 3 else 40.0 if i < 2 * n // 3 else 20.0))
                    for i in range(n)]
        assert segment_regions(frames(29)) == ([], 0.0)
        bounds, conf = segment_regions(frames(30))
        assert len(bounds) == 2 and conf > 0.5

    def test_overview_decimates_only_when_strictly_over_twice_the_target(self, monkeypatch):
        class FA:
            def __init__(self, thresholds=None):
                self.th = Thresholds()

            def analyze(self, f):
                return res(f.index, f.timestamp), np.zeros((16, 16), np.uint8)

        monkeypatch.setattr(analysis, "FrameAnalyzer", FA)
        monkeypatch.setattr(analysis, "OVERVIEW_TARGET", 2)
        frames = lambda n: (Frame(i, i * 0.5, uniform((i, i, i), 16)) for i in range(n))  # noqa: E731
        assert sorted(analyze_stream(frames(4)).overview) == [0, 1, 2, 3]      # 4 == 2 * target: kept
        assert sorted(analyze_stream(frames(5)).overview) == [0, 2, 4]         # 5 > 4: stride doubles

    def test_highlights_sample_every_frame_when_few(self, tmp_path):
        ev = [Event("bleeding", 0, 4, 0.0, 4.0, 2, 0.2, 5, "colon")]
        annotated = {i: uniform((i * 10, 50, 50), 100) for i in range(5)}
        st = Study("s", [res(i) for i in range(5)], ev, [], 0.0, annotated, {})
        assert report.write_highlights(st, tmp_path / "h.mp4", fps=4.0, per_event=24) == 4 + 5

    def test_timeline_blood_axis_floor(self):
        rs = [res(i, float(i), bleeding_score=0.05 if i == 4 else 0.0) for i in range(10)]
        svg = report._timeline_svg(Study("s", rs, [], [], 0.0, {}, {}))
        poly = svg.split('<polyline points="')[1].split('"')[0]
        ys = [float(p.split(",")[1]) for p in poly.split()]
        assert min(ys) == 75.0 and max(ys) == 120.0     # 0.05 on a 0.1 floor -> half of the 90 px plot

    def test_synth_cli_defaults(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(synth, "write_video", lambda *a: seen.setdefault("a", a) and "out.mp4")
        assert cli.main(["synth"]) == 0
        assert seen["a"] == ("synthetic/capsule.mp4", 600, 256, 4.0, 7, False)

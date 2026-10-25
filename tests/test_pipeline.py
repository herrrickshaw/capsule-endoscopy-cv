import socket
import struct
import threading

import numpy as np
import pytest

from capsule_cv.analysis import FrameAnalyzer, analyze_stream
from capsule_cv.ingest import Frame, Receiver, encode_message, open_source, send_frames
from capsule_cv.report import write_all
from capsule_cv.synth import SyntheticCapsule, write_video


@pytest.fixture(scope="module")
def synthetic():
    cap = SyntheticCapsule(n_frames=300, size=256, fps=4.0, seed=3)
    return cap, analyze_stream(cap.frames(), source="synthetic")


def _overlaps(event, span):
    a, b = span
    return event.start_index < b and event.end_index >= a


def test_every_bleeding_episode_is_found(synthetic):
    cap, study = synthetic
    bleeds = [e for e in study.events if e.kind == "bleeding"]
    for span in cap.truth["bleeding"]:
        assert any(_overlaps(e, span) for e in bleeds), span
    assert len(bleeds) == len(cap.truth["bleeding"])  # no spurious episodes


def test_lesions_found_with_right_kind(synthetic):
    cap, study = synthetic
    for kind in ("pale", "red"):
        span = cap.truth[f"lesion_{kind}"][0]
        hits = [r for r in study.results[span[0]:span[1]] if any(l["kind"] == kind for l in r.lesions)]
        assert len(hits) >= (span[1] - span[0]) // 2, kind
    lesion_events = [e for e in study.events if e.kind == "lesion"]
    truth = cap.truth["lesion_pale"] + cap.truth["lesion_red"]
    assert all(any(_overlaps(e, s) for s in truth) for e in lesion_events)  # no false events


def test_blurred_frames_marked_unusable(synthetic):
    cap, study = synthetic
    blurred = [i for a, b in cap.truth["blur"] for i in range(a, b)]
    assert sum(not study.results[i].usable for i in blurred) >= 0.9 * len(blurred)


def test_region_boundaries_and_transit(synthetic):
    cap, study = synthetic
    truth_s = cap.ground_truth()["region_boundaries_s"]
    tol = 0.05 * cap.n / cap.fps
    assert len(study.boundaries) == 2
    assert all(abs(a - b) <= tol for a, b in zip(study.boundaries, truth_s))
    assert study.summary()["gastric_transit_s"] == pytest.approx(truth_s[0], abs=tol)


def test_dwell_frames_are_duplicates(synthetic):
    cap, study = synthetic
    a, b = cap.truth["dwell"][0]
    assert sum(r.duplicate for r in study.results[a + 1:b]) >= 0.9 * (b - a - 1)


def test_identical_frames_deduplicated():
    img = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    fa = FrameAnalyzer()
    flags = [fa.analyze(Frame(i, i * 0.5, img))[0].duplicate for i in range(3)]
    assert flags == [False, True, True]


def test_receiver_detects_loss_and_corruption(tmp_path):
    rx = Receiver("127.0.0.1", 0, archive_dir=tmp_path / "raw", accept_timeout=10)
    img = np.full((64, 64, 3), 120, np.uint8)

    def sender():
        with socket.create_connection(("127.0.0.1", rx.port)) as s:
            s.sendall(encode_message(0, 0.0, img))
            s.sendall(encode_message(1, 0.5, img))
            bad = bytearray(encode_message(2, 1.0, img))
            bad[-5] ^= 0xFF  # flip payload bits -> CRC failure
            s.sendall(bytes(bad))
            s.sendall(encode_message(5, 2.5, img))  # seq 3, 4 lost in transit

    t = threading.Thread(target=sender)
    t.start()
    frames = list(rx.frames())
    t.join()
    assert [f.timestamp for f in frames] == [0.0, 0.5, 2.5]
    assert (rx.stats.received, rx.stats.corrupted, rx.stats.lost) == (3, 1, 2)
    assert rx.stats.gaps == [(3, 4)]
    assert len(list((tmp_path / "raw").iterdir())) == 3


def test_video_to_report_over_tcp(tmp_path):
    video = write_video(tmp_path / "cap.mp4", n_frames=200, seed=5)
    rx = Receiver("127.0.0.1", 0, accept_timeout=10)
    t = threading.Thread(target=send_frames, args=(open_source(video), "127.0.0.1", rx.port))
    t.start()
    study = analyze_stream(rx.frames(), source="tcp")
    t.join()
    assert len(study.results) == 200
    # blur must still be caught after MPEG + JPEG compression (block edges fake sharpness)
    blurred = [i for a, b in SyntheticCapsule(200, seed=5).truth["blur"] for i in range(a, b)]
    assert sum(not study.results[i].usable for i in blurred) >= 0.9 * len(blurred)
    assert sum(r.usable for r in study.results) >= 0.85 * 200
    paths = write_all(study, tmp_path / "out")
    html = paths["html"].read_text()
    assert "Decision-support only" in html and "<svg" in html
    assert paths["video"].stat().st_size > 1000
    assert paths["csv"].read_text().count("\n") == 201

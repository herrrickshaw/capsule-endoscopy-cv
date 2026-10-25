"""Command line: analyze recordings, receive a live capsule stream, simulate a sender, or make test data."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .analysis import analyze_stream
from .ingest import Receiver, open_source, send_frames
from .report import write_all


def _report(study, out) -> None:
    paths = write_all(study, out)
    print(json.dumps(study.summary(), indent=2))
    for k, p in paths.items():
        print(f"{k:>5}: {p}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="capsule_cv", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyse a recorded video or folder of frames")
    a.add_argument("input")
    a.add_argument("--out", default="capsule_report")
    a.add_argument("--fps", type=float, help="capture rate if the container doesn't record it (frames/s)")

    r = sub.add_parser("receive", help="receive frames from the capsule recorder over TCP, then analyse")
    r.add_argument("--host", default="0.0.0.0")
    r.add_argument("--port", type=int, default=5555)
    r.add_argument("--archive", default="session_frames", help="where raw received JPEGs are kept")
    r.add_argument("--out", default="capsule_report")

    s = sub.add_parser("send", help="simulate the recorder: stream a video/folder to a receiver")
    s.add_argument("input")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5555)
    s.add_argument("--rate", type=float, help="throttle to N frames/s (real capsules send ~2-6)")
    s.add_argument("--fps", type=float)

    g = sub.add_parser("synth", help="generate a synthetic capsule video with ground truth")
    g.add_argument("--out", default="synthetic/capsule.mp4")
    g.add_argument("--frames", type=int, default=600)
    g.add_argument("--size", type=int, default=256)
    g.add_argument("--fps", type=float, default=4.0)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--images", action="store_true", help="write a PNG folder instead of an mp4")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "analyze":
        _report(analyze_stream(open_source(args.input, args.fps), source=args.input, progress_every=1000), args.out)
    elif args.cmd == "receive":
        rx = Receiver(args.host, args.port, archive_dir=args.archive)
        print(f"listening on {args.host}:{rx.port} ...", flush=True)
        study = analyze_stream(rx.frames(), source=f"tcp://{args.host}:{rx.port}", progress_every=1000)
        st = rx.stats
        print(f"received={st.received} lost={st.lost} corrupted={st.corrupted} bytes={st.bytes}")
        _report(study, args.out)
    elif args.cmd == "send":
        n = send_frames(open_source(args.input, args.fps), args.host, args.port, args.rate)
        print(f"sent {n} frames")
    elif args.cmd == "synth":
        from .synth import write_video
        print(write_video(args.out, args.frames, args.size, args.fps, args.seed, args.images))
    return 0


if __name__ == "__main__":
    sys.exit(main())

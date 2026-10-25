"""Frame sources: recorded videos, image folders, and a live TCP link from the capsule's data recorder.

Wire protocol (one message per frame, big-endian):

    magic   4s   b"CAPS"
    seq     I    frame sequence number assigned by the capsule (gaps = lost frames)
    ts      d    seconds since capsule activation
    length  I    JPEG payload length in bytes
    crc32   I    CRC32 of the payload
    payload      JPEG-encoded frame

A real capsule streams over RF to a body-worn recorder; this models the recorder -> workstation hop,
where frames arrive as compressed images with sequence numbers and checksums.
"""
from __future__ import annotations

import logging
import socket
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAGIC = b"CAPS"
HEADER = struct.Struct("!4sIdII")


@dataclass
class Frame:
    index: int
    timestamp: float  # seconds since capsule activation
    image: np.ndarray  # BGR uint8


def iter_video(path: str | Path, fps: float | None = None) -> Iterator[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    rate = fps or cap.get(cv2.CAP_PROP_FPS) or 2.0
    i = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield Frame(i, i / rate, img)
            i += 1
    finally:
        cap.release()


def iter_image_dir(path: str | Path, fps: float = 2.0) -> Iterator[Frame]:
    files = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for i, p in enumerate(files):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("skipping unreadable image %s", p)
            continue
        yield Frame(i, i / fps, img)


def open_source(path: str | Path, fps: float | None = None) -> Iterator[Frame]:
    p = Path(path)
    if p.is_dir():
        return iter_image_dir(p, fps or 2.0)
    return iter_video(p, fps)


def encode_message(seq: int, timestamp: float, image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    payload = buf.tobytes()
    return HEADER.pack(MAGIC, seq, timestamp, len(payload), zlib.crc32(payload)) + payload


def send_frames(frames: Iterable[Frame], host: str, port: int, rate: float | None = None) -> int:
    """Simulate the recorder pushing frames to the workstation. `rate` throttles to N frames/s."""
    sent = 0
    with socket.create_connection((host, port)) as sock:
        for f in frames:
            sock.sendall(encode_message(f.index, f.timestamp, f.image))
            sent += 1
            if rate:
                time.sleep(1.0 / rate)
    return sent


@dataclass
class ReceiverStats:
    received: int = 0
    corrupted: int = 0
    lost: int = 0  # inferred from sequence gaps
    bytes: int = 0
    gaps: list[tuple[int, int]] = field(default_factory=list)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    chunks, remaining = [], n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class Receiver:
    """TCP receiver for one capsule session. Bind happens in __init__ so `port` is known before frames()."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5555, archive_dir: str | Path | None = None,
                 accept_timeout: float | None = None):
        self.stats = ReceiverStats()
        self.archive_dir = Path(archive_dir) if archive_dir else None
        if self.archive_dir:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(1)
        self._server.settimeout(accept_timeout)

    @property
    def port(self) -> int:
        return self._server.getsockname()[1]

    def frames(self) -> Iterator[Frame]:
        conn, addr = self._server.accept()
        log.info("capsule recorder connected from %s", addr)
        expected = None
        out_index = 0
        with conn:
            while True:
                header = _recv_exact(conn, HEADER.size)
                if header is None:
                    break
                magic, seq, ts, length, crc = HEADER.unpack(header)
                if magic != MAGIC:
                    raise ValueError(f"bad magic {magic!r}: stream desynchronised")
                payload = _recv_exact(conn, length)
                if payload is None:
                    break
                self.stats.bytes += HEADER.size + length
                if expected is not None and seq > expected:
                    self.stats.lost += seq - expected
                    self.stats.gaps.append((expected, seq - 1))
                expected = seq + 1
                if zlib.crc32(payload) != crc:
                    self.stats.corrupted += 1
                    log.warning("frame seq=%d failed CRC, dropped", seq)
                    continue
                img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    self.stats.corrupted += 1
                    continue
                if self.archive_dir:
                    (self.archive_dir / f"frame_{seq:07d}.jpg").write_bytes(payload)
                self.stats.received += 1
                yield Frame(out_index, ts, img)
                out_index += 1
        self._server.close()

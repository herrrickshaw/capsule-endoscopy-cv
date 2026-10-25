# capsule_cv — capsule endoscopy video analysis & reporting

Computer-vision triage for wireless capsule endoscopy ("pill cam"). The capsule records the GI tract at
2–6 frames/s for 8–12 h, producing roughly 50–100k frames. Frames reach a workstation either live or as a
recording. The tool screens every frame and hands the reader a short list of where to look.

> **Research / decision-support only.** Classical colour and texture heuristics. Not a medical device,
> not validated on patient data, and not a diagnosis. Absence of a flag does not exclude pathology.

## Pipeline

```
capsule ──RF──> body-worn recorder ──TCP (ingest.Receiver)──┐
recorded .mp4/.avi or folder of frames (ingest.open_source) ┴─> analysis.FrameAnalyzer (per frame)
                                                                 └─> analyze_stream -> Study
                                                                      └─> report.write_all
```

| Stage | What it does |
|---|---|
| **Ingest** | Video and image-folder readers. A TCP receiver with sequence numbers and CRC32 per frame counts lost and corrupted frames and archives the raw JPEGs. |
| **Quality** | Brightness, over-exposure, bile/debris fraction, and a no-reference blur score (Crété-Roffet re-blur, at half resolution so JPEG/MPEG blocks don't fake sharpness). |
| **Suspected blood** | Saturated red hue *and* R/G ≥ 2.2. The ratio is what separates blood from pink mucosa. Score = fraction of the field of view. |
| **Lesion candidates** | Compact spots that stand out from a median-filtered *local* background: `pale` (ulcer/fibrin-like) and `red` (angioectasia-like). Tissue next to blood or debris is excluded. |
| **Redundancy** | Histogram correlation plus pixel difference against the last *kept* frame, which drops dwell periods from the reading list. |
| **Regions** | Two-change-point segmentation of colour/texture gives stomach / small bowel / colon boundaries, plus gastric and small-bowel transit times with a fit confidence. |
| **Events** | Flagged frames grouped into episodes (gap ≤ 3 s, ≥ 2 frames), each with a peak frame and region. |

Outputs in `--out`: `report.html` (self-contained: KPIs, timeline, events with annotated peak frames,
landmarks, overview gallery), `highlights.mp4` (title card plus annotated frames per event),
`frames.csv` (every metric for every frame), `study.json`.

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# make a synthetic study with ground truth, then analyse it
.venv/bin/python -m capsule_cv synth --out synthetic/capsule.mp4
.venv/bin/python -m capsule_cv analyze synthetic/capsule.mp4 --out report

# live: workstation listens, recorder (simulated here) streams
.venv/bin/python -m capsule_cv receive --port 5555 --archive session_frames --out report
.venv/bin/python -m capsule_cv send synthetic/capsule.mp4 --port 5555 --rate 4

.venv/bin/python -m pytest -q tests
```

Real recordings exported as frame folders: `analyze path/to/frames --fps 2`.

## Wire protocol (recorder → workstation)

Big-endian header `!4sIdII` = magic `CAPS`, sequence number, timestamp (s since activation),
payload length, CRC32. The JPEG payload follows. Sequence gaps count as lost frames. CRC failures are
dropped and counted.

## Clinical context

[docs/LITERATURE.md](docs/LITERATURE.md) reviews guideline indications, result types and grading
(Saurin P0–P3, Lewis/CECDAI, SPICE, cleansing scales) and published AI benchmarks. It maps each
result type to what this tool does and does not detect.

## Calibration and limits

All thresholds live in `analysis.Thresholds`. They were tuned on the synthetic generator, which has
exact ground truth (all events and both region boundaries recovered exactly). **Re-tune them on
labelled real data before trusting any number.** The public Kvasir-Capsule dataset (Smedsrud et al.,
*Scientific Data* 2021) is a reasonable start. Known gaps: no polyp or tumour detection, no deep model,
and region segmentation assumes the study reaches the colon. To add a learned detector, call it inside
`FrameAnalyzer.analyze` and append to `lesions`. Events, the report and the video then pick it up unchanged.

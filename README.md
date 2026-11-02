# capsule_cv: capsule endoscopy video triage and reporting

Computer-vision triage for wireless capsule endoscopy ("pill cam"). The capsule records the GI tract at
2-6 frames/s for 8-12 h, producing roughly 50-100k frames, delivered live over TCP or as a recording or frame
folder. `capsule_cv` screens every frame and gives the reader a short list of where to look. Output is a
self-contained HTML report, a highlights video, a per-frame CSV and a study JSON.

> Status: research-only. Classical colour and texture heuristics; not a medical device, not validated on patient
> data, and not a diagnosis. Absence of a flag does not exclude pathology. All tests and demos use synthetic
> frames; no patient data or model weights are included. No licence file yet (all rights reserved).

## What it does
- Reads `.mp4`/`.avi` recordings or image folders, or receives frames over TCP with sequence numbers and CRC32.
- Scores frame quality: brightness, over-exposure, bile/debris fraction and a no-reference blur score.
- Flags suspected blood (saturated red hue and R/G >= 2.2) and focal `pale` / `red` lesion candidates.
- Drops near-duplicate frames (dwell periods) from the reading list.
- Segments stomach / small bowel / colon with two change points and estimates gastric and small-bowel transit.
- Groups flagged frames into events (gap <= 3 s, >= 2 frames), each with a peak frame and region.

## Quick start
```bash
git clone https://github.com/herrrickshaw/capsule-endoscopy-cv.git
cd capsule-endoscopy-cv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m capsule_cv synth --out synthetic/capsule.mp4     # synthetic study with ground truth
python -m capsule_cv analyze synthetic/capsule.mp4 --out report
```

Live link (workstation listens, recorder simulated):
```bash
python -m capsule_cv receive --port 5555 --archive session_frames --out report
python -m capsule_cv send synthetic/capsule.mp4 --port 5555 --rate 4
```
Real recordings exported as frame folders: `python -m capsule_cv analyze path/to/frames --fps 2`.

## How it works
```
capsule --RF--> body-worn recorder --TCP (ingest.Receiver)--+
recorded .mp4/.avi or frame folder (ingest.open_source) ----+--> analysis.FrameAnalyzer (per frame)
                                                                  --> analysis.analyze_stream -> Study
                                                                       --> report.write_all
```

| Stage | Module | Output |
|---|---|---|
| Ingest | `ingest.py` | `Frame` stream; TCP receiver counts lost and corrupted frames and archives raw JPEGs |
| Quality, blood, lesions, redundancy | `analysis.FrameAnalyzer` (`_quality`, `_blood`, `_lesions`, `_is_duplicate`) | `FrameResult` per frame |
| Regions and events | `analysis.segment_regions`, `group_events` | boundaries, transit times, `Event` list |
| Report | `report.py` | `report.html`, `highlights.mp4`, `frames.csv`, `study.json` |
| Test data | `synth.py` | synthetic video with ground truth |

Design notes: blur uses the Crete-Roffet re-blur score at half resolution so JPEG/MPEG blocks cannot fake
sharpness. The R/G ratio is what separates blood from pink mucosa. Lesion candidates are measured against a
median-filtered *local* background, and tissue next to blood or debris is excluded. All thresholds live in
`analysis.Thresholds`.

### Wire protocol (recorder to workstation)
Big-endian header `!4sIdII`: magic `CAPS`, sequence number, timestamp (s since activation), payload length,
CRC32; the JPEG payload follows. Sequence gaps count as lost frames. CRC failures are dropped and counted.

## Repository layout
| Path | Purpose |
|---|---|
| `capsule_cv/` | package: `ingest`, `analysis`, `report`, `synth`, `cli` (`python -m capsule_cv`) |
| `tests/` | pytest: pipeline tests on synthetic ground truth, unit and CLI smoke tests |
| `docs/LITERATURE.md` | clinical context: guideline indications, grading scales (Saurin P0-P3, Lewis/CECDAI, SPICE, cleansing scales), published AI benchmarks, and what this tool does and does not detect |
| `pyproject.toml` | ruff and pytest configuration |

## Configuration
No environment variables and no secrets. Detector thresholds are the `analysis.Thresholds` dataclass; pass an
instance to `analyze_stream(..., thresholds=...)`. CLI flags: `analyze --out --fps`, `receive --host --port
--archive --out`, `send --host --port --rate --fps`, `synth --out --frames --size --fps --seed --images`.

## Testing
```bash
pip install -r requirements.txt ruff
ruff check .
pytest -q
```

## Limitations and known issues
- Thresholds were tuned on the synthetic generator (exact ground truth: all events and both region boundaries
  recovered). **Re-tune them on labelled real data before trusting any number.** The public Kvasir-Capsule
  dataset (Smedsrud et al., *Scientific Data* 2021) is a reasonable start.
- No polyp or tumour detection and no deep model; region segmentation assumes the study reaches the colon.
- To add a learned detector, call it inside `FrameAnalyzer.analyze` and append to `lesions`; events, report and
  video pick it up unchanged.

## Related repositories
External projects (not part of this account):

Datasets for validating or training detectors:
- [simula/kvasir-capsule](https://github.com/simula/kvasir-capsule): official Kvasir-Capsule repo (117 videos, 47,238 labelled frames, 14 classes).
- [EKFZ-AI-Endoscopy/GalarCapsuleML](https://github.com/EKFZ-AI-Endoscopy/GalarCapsuleML): code for the Galar dataset (videos labelled per frame, with several labels allowed per frame).
- [MECLabTUDA/kvasir-capsule-loader](https://github.com/MECLabTUDA/kvasir-capsule-loader): unofficial Python loader for Kvasir-Capsule.

Comparable systems and models:
- [CUHK-AIM-Group/CE-Reporter](https://github.com/CUHK-AIM-Group/CE-Reporter): turns a capsule video into a written report using a vision-language model (LLaVA-based; needs a GPU).
- [abhijeetdhupia/WCE-Pipeline](https://github.com/abhijeetdhupia/WCE-Pipeline): sorts frames into normal and abnormal, then segments the abnormal ones (trained on the KID dataset).
- [mvrcii/capsule_vision_challenge_2024](https://github.com/mvrcii/capsule_vision_challenge_2024): 3rd place in Capsule Vision 2024; adapts a pre-trained foundation model to endoscopy images.
- [MAILAB-Yonsei/capsule_endoscopy_detection](https://github.com/MAILAB-Yonsei/capsule_endoscopy_detection): lesion detection that combines 11 detector models.

Simulation and image enhancement:
- [CapsuleEndoscope/VirtualCapsuleEndoscopy](https://github.com/CapsuleEndoscope/VirtualCapsuleEndoscopy): VR-Caps, a Unity simulator that generates synthetic capsule data. It is more realistic than `capsule_cv.synth`.
- [longbai1006/LLCaps](https://github.com/longbai1006/LLCaps): enhancement for dark (low-light) capsule images (MICCAI 2023).

## Licence
No licence file yet; all rights reserved.

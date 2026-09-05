# Real-Time Indian Sign Language (ISL) → Speech Translation

> Real-time recognition of a controlled ISL vocabulary from webcam input,
> with English sentence generation and speech output.

**Status:** Phase 2 — Dataset Collection Pipeline

---

## Tech Stack

| Component        | Technology                                  |
| ---------------- | ------------------------------------------- |
| Hand tracking    | MediaPipe Hand Landmarker (1.0.1)           |
| Camera           | OpenCV                                      |
| ML framework     | NumPy (data pipeline), PyTorch/sklearn (TBD)|
| Language         | Python 3.12                                 |
| OS               | Windows (development)                       |

## Project Structure

```
sign-to-speech/
├── ml/
│   ├── hand_tracking.py      # Phase 1 — standalone landmark demo
│   ├── collect_data.py        # Phase 2 — sequence dataset recorder
│   ├── preprocessing.py       # Landmark normalization
│   ├── config.py              # Centralized configuration
│   ├── verify_dataset.py      # Dataset validation
│   └── hand_landmarker.task   # MediaPipe model (NOT committed — see below)
├── dataset/                   # Recorded .npy samples (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd sign-to-speech
python -m venv .venv
.venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the MediaPipe model

The `hand_landmarker.task` file (~7.8 MB) is **intentionally not committed**
to the repository to keep it lightweight.

Download it manually:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Place the file in the `ml/` directory:

```
ml/hand_landmarker.task
```

### 4. Verify hand tracking works

```bash
python ml/hand_tracking.py
```

You should see a webcam preview with green landmark dots on detected hands.
Press `Q` to quit.

## Dataset Collection

### Recording sign samples

```bash
python -m ml.collect_data
```

1. Enter a **subject/participant ID** (e.g. `s01`).
2. Enter a **sign name** (e.g. `HELLO`, `WATER`, `NEUTRAL`).
3. In the OpenCV window:
   - **`R`** — Start recording (3-second countdown, then captures 30 frames).
   - **`N`** — Change sign name.
   - **`P`** — Change subject ID.
   - **`Q`** — Quit.

Each sample is saved as:

```
dataset/<SIGN_NAME>/<subject_id>_sample_<NNN>.npy
Shape: (30, 126)   # 30 frames × 126 features (63 per hand × 2 hands)
```

### The NEUTRAL class

Record `NEUTRAL` samples (hands at rest / no sign) so the model can
eventually distinguish "idle" from an active sign.

### Subject IDs

Filenames include the subject ID to support **per-subject train/test splits**.
This prevents data leakage from the same person appearing in both training
and evaluation sets.

### Verifying recorded data

```bash
python -m ml.verify_dataset
```

Reports class counts, per-subject breakdowns, shape validation, and feature
statistics. No webcam required.

## Landmark Preprocessing

The preprocessing pipeline (see `ml/preprocessing.py`) applies:

1. **Wrist-relative translation** — subtracts wrist position (landmark 0).
2. **Scale normalization** — divides by wrist-to-middle-finger-MCP distance.
3. **Missing-hand zero-padding** — consistent 126-dim vector even with 0 or 1 hands.

**Intentionally NOT applied:** rotation normalization (would destroy orientation
information important for many ISL signs).

## Handedness Note

The collector mirrors each camera frame before both preview and MediaPipe
detection. MediaPipe's handedness model expects that selfie-style input, so
the vector is consistently `[physical left 63] + [physical right 63]`.
Before recording the first samples, raise one physical hand at a time and
confirm the on-screen label: physical left must say `Left`; physical right
must say `Right`. Keep `MIRROR_CAMERA = True` for every recording in one
dataset.

MediaPipe labels hands as "Left" / "Right" **from the image's perspective**,
which may be the mirror of the signer's physical hand depending on camera
setup. The collector displays colour-coded labels (gold = Left, cyan = Right)
so you can verify with your camera. Consistency across recordings is what
matters for model training.

## Limitations

- This is a **controlled vocabulary** recognizer, not a full ISL translator.
- ISL grammar (word order, spatial references, non-manual markers) is not yet
  handled — the current system recognizes individual signs.
- The model has not been trained yet (Phase 2 = data collection only).

## Future Roadmap

| Phase | Description                      |
| ----- | -------------------------------- |
| 1 ✅  | Hand tracking                    |
| 2 🔄  | Dataset collection pipeline      |
| 3     | Baseline sign classifier         |
| 4     | Temporal / sequence recognition  |
| 5     | Continuous real-time detection   |
| 6     | ISL → English language layer     |
| 7     | Text-to-Speech                   |
| 8     | Backend API (FastAPI)            |
| 9     | Frontend UI (Next.js)            |
| 10    | Integration & testing            |

## License

TBD

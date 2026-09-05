"""
Centralized configuration for the ISL Sign Language project.

All constants and paths are defined here so they can be changed in one
place.  Nothing in this file is a permanent assumption — values like
SEQUENCE_LENGTH are configurable baselines that should be tuned once
real data is collected and evaluated.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
MODELS_DIR = PROJECT_ROOT / "models"          # trained checkpoints (future)

# ──────────────────────────────────────────────────────────────────────
# Landmark representation
# ──────────────────────────────────────────────────────────────────────

NUM_HANDS = 2                                  # max hands MediaPipe detects
NUM_LANDMARKS = 21                             # MediaPipe hand landmarks per hand
NUM_COORDS = 3                                 # x, y, z

FEATURES_PER_HAND = NUM_LANDMARKS * NUM_COORDS  # 21 × 3 = 63
TOTAL_FEATURES = FEATURES_PER_HAND * NUM_HANDS  # 63 × 2 = 126

# Feature vector layout:
#   [0:63]   → left  hand (21 landmarks × 3 coords)
#   [63:126] → right hand (21 landmarks × 3 coords)
# Missing hands are zero-padded.

# ──────────────────────────────────────────────────────────────────────
# Sequence recording
# ──────────────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 30   # frames per sample — configurable baseline (~1 s at 30 FPS)
COUNTDOWN_SECONDS = 3  # countdown before recording starts

# ──────────────────────────────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────────────────────────────

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
# MediaPipe's handedness model assumes mirrored (selfie-style) input. Keep
# detector input and preview mirrored so slots refer to the signer's hands.
MIRROR_CAMERA = True

# ──────────────────────────────────────────────────────────────────────
# MediaPipe detection thresholds
# ──────────────────────────────────────────────────────────────────────

MIN_DETECTION_CONFIDENCE = 0.6
MIN_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.7

# ──────────────────────────────────────────────────────────────────────
# Dataset collection
# ──────────────────────────────────────────────────────────────────────

DEFAULT_SUBJECT_ID = "s01"

# The NEUTRAL class name — used to record "no sign" baseline data so the
# real-time system can distinguish idle from an active sign.
NEUTRAL_CLASS = "NEUTRAL"

"""
ISL Dataset Collector — Sequence Recording Tool.

Records sequences of normalized hand landmarks for ISL sign recognition
training.  Each sample is a fixed-length sequence of 126-dimensional feature
vectors saved as a ``.npy`` file.

Usage
─────
    python -m ml.collect_data

Controls (in the OpenCV window)
───────────────────────────────
    R — Start recording a sample (countdown → capture → save)
    N — Change current sign/class name
    P — Change participant / subject ID
    Q — Quit

Saved file format
─────────────────
    dataset/<SIGN_NAME>/<subject_id>_sample_<NNN>.npy
    Shape : (SEQUENCE_LENGTH, TOTAL_FEATURES) = (30, 126) by default

The NEUTRAL class is explicitly supported — use it to record "no-sign"
baseline data so the eventual real-time system can distinguish idle from
an active sign.

Subject / participant IDs
─────────────────────────
Filenames embed the subject ID (e.g. ``s01_sample_001.npy``) so that
train/test splits can be done per-subject rather than randomly, avoiding
data leakage from the same person appearing in both splits.
"""

from __future__ import annotations

import cv2
import sys
import time
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ml.config import (
    MODEL_PATH,
    DATASET_DIR,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    MIRROR_CAMERA,
    MIN_DETECTION_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    DEFAULT_SUBJECT_ID,
    COUNTDOWN_SECONDS,
    NEUTRAL_CLASS,
)
from ml.preprocessing import build_feature_vector


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# MediaPipe Hand Landmarker
# ──────────────────────────────────────────────────────────────────────
# Initialization uses model_asset_buffer (in-memory bytes) — the same
# pattern that was already working in the original hand_tracking.py.

def _create_detector() -> vision.HandLandmarker:
    """Create and return a MediaPipe HandLandmarker instance."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found at {MODEL_PATH}\n"
            "Download hand_landmarker.task from:\n"
            "  https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/"
            "hand_landmarker.task\n"
            "and place it in the ml/ directory."
        )

    base_options = python.BaseOptions(
        model_asset_buffer=MODEL_PATH.read_bytes()
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


# ──────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────

# Colour coding:
#   Gold   (255, 200, 0)  → MediaPipe "Left"  hand label
#   Cyan   (0, 200, 255)  → MediaPipe "Right" hand label
# These are the labels AS SEEN IN THE IMAGE, which may be the mirror of
# the signer's physical hand (see preprocessing.py docstring for details).

_COLOUR_LEFT = (0, 200, 255)   # gold in BGR
_COLOUR_RIGHT = (255, 200, 0)  # cyan in BGR

# MediaPipe hand skeleton connections (parent → child)
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]


def _draw_landmarks(frame: np.ndarray, result) -> None:
    """Draw hand landmarks, skeleton, and handedness labels."""
    h, w, _ = frame.shape

    for hand_landmarks, handedness in zip(
        result.hand_landmarks, result.handedness
    ):
        label = handedness[0].category_name
        score = handedness[0].score
        colour = _COLOUR_LEFT if label == "Left" else _COLOUR_RIGHT

        # Pixel positions for each landmark
        pts = []
        for lm in hand_landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            pts.append((px, py))
            cv2.circle(frame, (px, py), 5, colour, -1)

        # Skeleton connections
        for a, b in _CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], colour, 2)

        # Handedness label near wrist
        wx, wy = pts[0]
        cv2.putText(
            frame,
            f"{label} ({score:.0%})",
            (wx - 40, wy - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
        )


def _draw_hud(
    frame: np.ndarray,
    sign_name: str,
    subject_id: str,
    sample_count: int,
    state: str,
    extra: str = "",
) -> None:
    """Draw the heads-up display (status bar + controls hint)."""
    h, w, _ = frame.shape

    # ── Semi-transparent bar at top ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Line 1: sign / subject / count
    cv2.putText(
        frame,
        f"Sign: {sign_name}  |  Subject: {subject_id}  |  Samples: {sample_count}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    # Line 2: state
    state_colours = {
        "IDLE": (100, 255, 100),
        "COUNTDOWN": (0, 255, 255),
        "RECORDING": (0, 0, 255),
    }
    colour = state_colours.get(state, (255, 255, 255))
    status_text = f"State: {state}"
    if extra:
        status_text += f"  —  {extra}"
    cv2.putText(frame, status_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

    # ── Controls hint at bottom ──
    hint = "[R] Record  [N] New sign  [P] Subject  [Q] Quit"
    cv2.putText(frame, hint, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # ── Red border during recording ──
    if state == "RECORDING":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)


# ──────────────────────────────────────────────────────────────────────
# Sample numbering
# ──────────────────────────────────────────────────────────────────────

def _safe_name(value: str, field_name: str) -> str:
    """Validate a class or subject identifier used in a dataset path."""
    value = value.strip()
    if not value or value in {".", ".."}:
        raise ValueError(f"{field_name} cannot be empty, '.' or '..'.")
    if any(char in value for char in '\\/:*?\"<>|'):
        raise ValueError(
            f"{field_name} cannot contain path separators or Windows-reserved characters."
        )
    return value


def _next_sample_number(sign_dir, subject_id: str) -> int:
    """Return the next unused sample number for this subject in this dir."""
    highest = 0
    for path in sign_dir.glob(f"{subject_id}_sample_*.npy"):
        try:
            highest = max(highest, int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return highest + 1


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Terminal setup ──
    print("=" * 60)
    print("  ISL Dataset Collector")
    print("=" * 60)
    print()
    print(f"  Sequence length : {SEQUENCE_LENGTH} frames")
    print(f"  Features/frame  : {TOTAL_FEATURES}")
    print(f"  NEUTRAL class   : '{NEUTRAL_CLASS}' (for no-sign baseline)")
    print()

    subject_id = _safe_name(
        input(f"  Subject/participant ID [{DEFAULT_SUBJECT_ID}]: ").strip()
        or DEFAULT_SUBJECT_ID,
        "Subject ID",
    )

    sign_name = input("  Sign name (e.g. HELLO, NEUTRAL): ").strip().upper()
    while not sign_name:
        sign_name = input("  Sign name cannot be empty: ").strip().upper()
    sign_name = _safe_name(sign_name, "Sign name")

    print()
    print(f"  → Subject : {subject_id}")
    print(f"  → Sign    : {sign_name}")
    print()

    # ── Create detector ──
    print("  Loading MediaPipe Hand Landmarker...")
    detector = _create_detector()
    print("  ✓ Model loaded")
    print()

    # ── Camera ──
    print("  Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Could not open webcam")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  ✓ Camera: {actual_w}×{actual_h} @ {actual_fps:.0f} FPS")
    print()
    print("  Controls: [R] Record  [N] New sign  [P] Change subject  [Q] Quit")
    print()

    # ── State ──
    state = "IDLE"
    countdown_start = 0.0
    recording_buffer: list[np.ndarray] = []
    frames_without_hands = 0

    sign_dir = DATASET_DIR / sign_name
    sign_dir.mkdir(parents=True, exist_ok=True)
    sample_count = len(list(sign_dir.glob(f"{subject_id}_sample_*.npy")))

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("  Could not read frame — exiting.")
                break

            # ── Detect ──
            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            # ── Draw skeleton ──
            _draw_landmarks(frame, result)

            # ── Build features (always, so we can display info) ──
            features, info = build_feature_vector(result)

            # ── State machine ──
            extra = ""

            if state == "COUNTDOWN":
                elapsed = time.time() - countdown_start
                remaining = COUNTDOWN_SECONDS - elapsed

                if remaining <= 0:
                    # Transition to recording
                    state = "RECORDING"
                    recording_buffer = []
                    frames_without_hands = 0
                else:
                    extra = f"Starting in {int(remaining) + 1}…"
                    # Big countdown number in center
                    ch, cw, _ = frame.shape
                    cv2.putText(
                        frame,
                        str(int(remaining) + 1),
                        (cw // 2 - 50, ch // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        5,
                        (0, 255, 255),
                        10,
                    )

            elif state == "RECORDING":
                recording_buffer.append(features.copy())
                if info["num_hands"] == 0:
                    frames_without_hands += 1

                frames_done = len(recording_buffer)
                extra = f"Frame {frames_done}/{SEQUENCE_LENGTH}"

                if frames_done >= SEQUENCE_LENGTH:
                    # ── Save ──
                    sequence = np.stack(recording_buffer, axis=0)
                    assert sequence.shape == (SEQUENCE_LENGTH, TOTAL_FEATURES), (
                        f"Expected ({SEQUENCE_LENGTH}, {TOTAL_FEATURES}), "
                        f"got {sequence.shape}"
                    )

                    sample_num = _next_sample_number(sign_dir, subject_id)
                    filename = f"{subject_id}_sample_{sample_num:03d}.npy"
                    filepath = sign_dir / filename
                    np.save(filepath, sequence)
                    sample_count += 1

                    # Console report
                    print(f"  ✓ Saved: {filepath.relative_to(DATASET_DIR.parent)}")
                    print(f"    Shape : {sequence.shape}  |  dtype: {sequence.dtype}")
                    print(
                        f"    Hands : {SEQUENCE_LENGTH - frames_without_hands}/"
                        f"{SEQUENCE_LENGTH} frames had ≥1 hand"
                    )
                    if frames_without_hands > SEQUENCE_LENGTH * 0.5:
                        print("    ⚠ WARNING: >50% of frames had no hands detected!")
                    print()

                    state = "IDLE"
                    recording_buffer = []

            # ── Draw HUD (always) ──
            _draw_hud(frame, sign_name, subject_id, sample_count, state, extra)

            # ── Show number of hands detected ──
            h_frame, w_frame, _ = frame.shape
            hands_text = f"Hands: {info['num_hands']}"
            if info["handedness_labels"]:
                hands_text += f"  ({', '.join(info['handedness_labels'])})"
            cv2.putText(
                frame,
                hands_text,
                (15, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            # ── Display ──
            cv2.imshow("ISL Dataset Collector", frame)

            # ── Keys ──
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r") and state == "IDLE":
                state = "COUNTDOWN"
                countdown_start = time.time()
                print(
                    f"  ● Recording '{sign_name}' (subject: {subject_id}) "
                    f"in {COUNTDOWN_SECONDS}s…"
                )

            elif key == ord("n") and state == "IDLE":
                # Pause camera display and ask in terminal
                cv2.destroyAllWindows()
                new_name = input("  Enter new sign name: ").strip().upper()
                if new_name:
                    try:
                        sign_name = _safe_name(new_name, "Sign name")
                    except ValueError as exc:
                        print(f"  [!] {exc}")
                        continue
                    sign_dir = DATASET_DIR / sign_name
                    sign_dir.mkdir(parents=True, exist_ok=True)
                    sample_count = len(
                        list(sign_dir.glob(f"{subject_id}_sample_*.npy"))
                    )
                    print(
                        f"  → Sign: {sign_name} "
                        f"(existing samples for {subject_id}: {sample_count})"
                    )
                else:
                    print(f"  → Keeping: {sign_name}")

            elif key == ord("p") and state == "IDLE":
                cv2.destroyAllWindows()
                new_subj = input(
                    f"  Enter new subject ID [{subject_id}]: "
                ).strip()
                if new_subj:
                    try:
                        subject_id = _safe_name(new_subj, "Subject ID")
                    except ValueError as exc:
                        print(f"  [!] {exc}")
                        continue
                    sample_count = len(
                        list(sign_dir.glob(f"{subject_id}_sample_*.npy"))
                    )
                    print(f"  → Subject: {subject_id}")
                else:
                    print(f"  → Keeping: {subject_id}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    # ── Summary ──
    print()
    print("=" * 60)
    print("  Session complete")
    print(f"  Subject       : {subject_id}")
    print(f"  Last sign     : {sign_name}")
    print(f"  Dataset dir   : {DATASET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

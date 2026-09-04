"""
Landmark preprocessing for ISL sign recognition.

Normalization pipeline
─────────────────────
1. **Wrist-relative translation** — subtract wrist position (landmark 0) from
   all 21 landmarks.  This makes features translation-invariant: the same sign
   is represented the same way regardless of where the hand appears in the frame.

2. **Scale normalization** — divide by the Euclidean distance from the wrist to
   the middle-finger MCP joint (landmark 9).  This makes features invariant to
   hand size and distance from the camera.

3. **Missing-hand padding** — if only one hand is detected, the other hand's 63
   features remain as zeros.

Design decisions (and what is intentionally NOT done)
─────────────────────────────────────────────────────
- **No rotation normalization.**  Many ISL signs are distinguished by hand
  orientation (palm facing up vs. down, fingers pointing left vs. right).
  Removing rotation would destroy this information.  Rotation normalization
  should only be introduced after empirical testing demonstrates it helps.

- **Z-depth is preserved.**  MediaPipe's z-coordinate encodes relative depth
  within the hand, which carries information about hand shape (e.g. fingers
  curling toward the camera vs. extending away).

- **Temporal information is preserved.**  Each frame is normalized
  independently; the normalization does NOT look at neighbouring frames or
  alter the temporal structure of a sequence.

MediaPipe handedness — how left/right ordering works
────────────────────────────────────────────────────
MediaPipe's ``handedness`` output classifies each detected hand as "Left" or
"Right".  The label refers to the hand **as seen in the image**, which is
typically the MIRROR of the signer's actual hand when using a standard
front-facing webcam (i.e. a signer's physical right hand often appears on the
left side of the non-mirrored camera image and MediaPipe may label it "Left").

**Important**: the exact mapping depends on camera mirroring settings.  The
``collect_data.py`` script draws colour-coded handedness labels on the preview
(gold = MediaPipe "Left", cyan = MediaPipe "Right") so you can empirically
verify which physical hand maps to which label with YOUR camera.

What matters for training is **consistency** — as long as the same physical
hand always occupies the same slot in the 126-dim feature vector, the model
can learn from it.

Feature vector layout
─────────────────────
    [  0 :  63 ]  →  left-hand landmarks  (21 × 3)
    [ 63 : 126 ]  →  right-hand landmarks (21 × 3)
"""

from __future__ import annotations

import numpy as np

from ml.config import (
    NUM_LANDMARKS,
    NUM_COORDS,
    FEATURES_PER_HAND,
    TOTAL_FEATURES,
)


# ──────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────

def extract_hand_landmarks(
    result,
) -> tuple[list[float], list[float], dict]:
    """Extract left/right hand landmarks from a MediaPipe result.

    Parameters
    ----------
    result
        A ``mediapipe.tasks.python.vision.HandLandmarkerResult`` returned by
        ``HandLandmarker.detect()``.

    Returns
    -------
    left_hand
        Flat list of 63 floats (21 landmarks × 3 coords).  All zeros when the
        left hand is not detected.
    right_hand
        Same format for the right hand.
    info
        Metadata dict::

            {
                "num_hands":         int,
                "left_detected":     bool,
                "right_detected":    bool,
                "handedness_labels": list[str],   # raw MediaPipe labels
                "handedness_scores": list[float], # confidence per label
            }
    """
    left_hand: list[float] = [0.0] * FEATURES_PER_HAND
    right_hand: list[float] = [0.0] * FEATURES_PER_HAND

    info: dict = {
        "num_hands": len(result.hand_landmarks),
        "left_detected": False,
        "right_detected": False,
        "handedness_labels": [],
        "handedness_scores": [],
    }

    for hand_landmarks, handedness in zip(
        result.hand_landmarks,
        result.handedness,
    ):
        coords: list[float] = []
        for lm in hand_landmarks:
            coords.extend([lm.x, lm.y, lm.z])

        label = handedness[0].category_name
        score = handedness[0].score
        info["handedness_labels"].append(label)
        info["handedness_scores"].append(score)

        if label == "Left":
            left_hand = coords
            info["left_detected"] = True
        elif label == "Right":
            right_hand = coords
            info["right_detected"] = True

    return left_hand, right_hand, info


# ──────────────────────────────────────────────────────────────────────
# Single-hand normalization
# ──────────────────────────────────────────────────────────────────────

def normalize_hand(landmarks: list[float]) -> list[float]:
    """Normalize a single hand's 63 raw landmark values.

    Steps
    -----
    1. Wrist-relative translation (subtract landmark 0).
    2. Scale by distance from wrist → middle-finger MCP (landmark 9).

    If the hand is all zeros (not detected), returns zeros unchanged.
    If the scale factor is near zero (degenerate detection), returns
    wrist-relative coordinates without scaling to avoid division by zero.

    Parameters
    ----------
    landmarks
        Flat list of 63 floats ``[x0, y0, z0, x1, y1, z1, …]``.

    Returns
    -------
    Normalized flat list of 63 floats.
    """
    # Fast path: missing hand
    if all(v == 0.0 for v in landmarks):
        return landmarks

    arr = np.array(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, NUM_COORDS)

    # Step 1 — wrist-relative translation
    wrist = arr[0].copy()
    arr = arr - wrist

    # Step 2 — scale by wrist-to-middle-MCP distance
    middle_mcp = arr[9]
    scale = float(np.linalg.norm(middle_mcp))

    if scale > 1e-6:
        arr = arr / scale

    return arr.flatten().tolist()


# ──────────────────────────────────────────────────────────────────────
# Full pipeline for one frame
# ──────────────────────────────────────────────────────────────────────

def build_feature_vector(
    result,
) -> tuple[np.ndarray, dict]:
    """Extract, normalize, and combine hand landmarks into a 126-dim vector.

    Parameters
    ----------
    result
        MediaPipe ``HandLandmarkerResult``.

    Returns
    -------
    features
        ``np.ndarray`` of shape ``(126,)`` with dtype ``float32``.
    info
        Metadata dict (see :func:`extract_hand_landmarks`).
    """
    left_raw, right_raw, info = extract_hand_landmarks(result)

    left_norm = normalize_hand(left_raw)
    right_norm = normalize_hand(right_raw)

    features = np.array(
        left_norm + right_norm,
        dtype=np.float32,
    )
    assert features.shape == (TOTAL_FEATURES,), (
        f"Expected shape ({TOTAL_FEATURES},), got {features.shape}"
    )

    return features, info


# ──────────────────────────────────────────────────────────────────────
# Sequence-level normalization (for re-processing saved raw data)
# ──────────────────────────────────────────────────────────────────────

def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Apply per-frame normalization to an entire sequence.

    Useful when re-normalizing data that was originally saved without
    normalization.  Each frame is normalized independently — temporal
    structure is preserved.

    Parameters
    ----------
    sequence
        ``np.ndarray`` of shape ``(seq_len, 126)`` with raw landmarks.

    Returns
    -------
    Normalized array of the same shape.
    """
    normalized = np.zeros_like(sequence)

    for i in range(sequence.shape[0]):
        frame = sequence[i]
        left_raw = frame[:FEATURES_PER_HAND].tolist()
        right_raw = frame[FEATURES_PER_HAND:].tolist()

        left_norm = normalize_hand(left_raw)
        right_norm = normalize_hand(right_raw)

        normalized[i] = np.array(left_norm + right_norm, dtype=np.float32)

    return normalized

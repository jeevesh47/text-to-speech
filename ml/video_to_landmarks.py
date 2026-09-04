"""
Convert pre-recorded sign-language videos into normalized landmark sequences.

Pipeline:
    .MOV/.MP4
        -> sample exactly 30 frames
        -> MediaPipe Hand Landmarker
        -> 21 landmarks per hand
        -> left/right slots (63 + 63)
        -> wrist-relative + scale normalization
        -> (30, 126) NumPy array
        -> dataset/<SIGN>/<SUBJECT>_sample_NNN.npy

Run from the project root:

    python -m ml.video_to_landmarks

Optional:

    python -m ml.video_to_landmarks --input data/adjectives
    python -m ml.video_to_landmarks --subject s01
    python -m ml.video_to_landmarks --limit 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .config import (
    MODEL_PATH,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    DATASET_DIR,
)
from .preprocessing import build_feature_vector


# ============================================================
# DEFAULT SETTINGS
# ============================================================

# Change this to the location of your downloaded dataset.
DEFAULT_INPUT_DIR = Path("adjectives_dataset")

# Subject ID used when the source dataset does not provide
# subject information.
DEFAULT_SUBJECT_ID = "s01"

# Set True ONLY if the source videos need horizontal mirroring
# to match the handedness convention used during training.
MIRROR_VIDEO = False

# Supported video formats.
VIDEO_EXTENSIONS = {
    ".mov",
    ".mp4",
    ".avi",
    ".mkv",
    ".webm",
}


# ============================================================
# MEDIAPIPE
# ============================================================

def create_detector() -> mp.tasks.vision.HandLandmarker:
    """
    Create MediaPipe Hand Landmarker in VIDEO mode.
    """

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_buffer=MODEL_PATH.read_bytes()
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    return HandLandmarker.create_from_options(options)


# ============================================================
# VIDEO HELPERS
# ============================================================

def get_video_frames(video_path: Path) -> list[np.ndarray]:
    """
    Read all frames from a video.

    Returns:
        List of BGR OpenCV frames.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        if MIRROR_VIDEO:
            frame = cv2.flip(frame, 1)

        frames.append(frame)

    cap.release()

    return frames


def sample_frames(
    frames: list[np.ndarray],
    sequence_length: int = SEQUENCE_LENGTH,
) -> list[np.ndarray]:
    """
    Sample exactly sequence_length frames from a video.

    Frames are selected evenly across the entire video.
    """

    if len(frames) == 0:
        raise ValueError("Video contains no readable frames.")

    if len(frames) == sequence_length:
        return frames

    indices = np.linspace(
        0,
        len(frames) - 1,
        sequence_length,
    ).round().astype(int)

    return [frames[i] for i in indices]


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def process_video(
    video_path: Path,
    detector: mp.tasks.vision.HandLandmarker,
) -> tuple[np.ndarray, float, float]:
    """
    Convert one video into a (30, 126) landmark sequence.

    Returns:
        sequence:
            Shape (30, 126)

        hand_coverage:
            Percentage of sampled frames containing at least one hand.

        left_coverage:
            Percentage of sampled frames containing a left hand.
    """

    frames = get_video_frames(video_path)

    original_frame_count = len(frames)

    sampled_frames = sample_frames(
        frames,
        SEQUENCE_LENGTH,
    )

    sequence = []

    frames_with_hand = 0
    frames_with_left_hand = 0

    for frame_index, frame in enumerate(sampled_frames):

        # OpenCV BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # MediaPipe VIDEO mode requires timestamps.
        timestamp_ms = frame_index * 33

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        # Existing preprocessing.py handles:
        # - left/right slots
        # - 21 landmarks
        # - wrist-relative normalization
        # - scale normalization
        # - zero padding
        features, info = build_feature_vector(result)

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        if features.shape != (TOTAL_FEATURES,):
            raise ValueError(
                f"Unexpected feature shape: "
                f"{features.shape}; expected "
                f"({TOTAL_FEATURES},)"
            )

        sequence.append(features)

        # Detection statistics
        if result.hand_landmarks:
            frames_with_hand += 1

            # Check MediaPipe handedness labels.
            for handedness in result.handedness:
                if handedness:
                    label = handedness[0].category_name

                    if label == "Left":
                        frames_with_left_hand += 1
                        break

    sequence = np.stack(sequence).astype(np.float32)

    expected_shape = (
        SEQUENCE_LENGTH,
        TOTAL_FEATURES,
    )

    if sequence.shape != expected_shape:
        raise ValueError(
            f"Unexpected sequence shape: "
            f"{sequence.shape}; expected "
            f"{expected_shape}"
        )

    hand_coverage = (
        frames_with_hand / SEQUENCE_LENGTH
    ) * 100.0

    left_coverage = (
        frames_with_left_hand / SEQUENCE_LENGTH
    ) * 100.0

    print(
        f"      source frames : {original_frame_count}"
    )
    print(
        f"      sampled frames: {SEQUENCE_LENGTH}"
    )
    print(
        f"      hand coverage : {hand_coverage:.1f}%"
    )
    print(
        f"      left coverage : {left_coverage:.1f}%"
    )

    return (
        sequence,
        hand_coverage,
        left_coverage,
    )


# ============================================================
# FILE MANAGEMENT
# ============================================================

def find_videos(input_dir: Path) -> list[Path]:
    """
    Find videos recursively.
    """

    videos = []

    for path in input_dir.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ):
            videos.append(path)

    return sorted(videos)


def get_sign_name(
    video_path: Path,
    input_root: Path,
) -> str:
    """
    Determine sign/class name from the video's parent folder.

    Example:

        adjectives_dataset/Beautiful/video1.mov

    becomes:

        Beautiful
    """

    relative_parent = video_path.parent.relative_to(input_root)

    if str(relative_parent) == ".":
        raise ValueError(
            f"Video must be inside a class/sign folder:\n"
            f"{video_path}"
        )

    # First directory below the dataset root is the class.
    sign_name = relative_parent.parts[0]

    return sign_name


def get_next_sample_number(
    output_dir: Path,
    subject_id: str,
) -> int:
    """
    Find the next available sample number.
    """

    existing = list(
        output_dir.glob(
            f"{subject_id}_sample_*.npy"
        )
    )

    if not existing:
        return 1

    numbers = []

    for path in existing:

        try:
            number_part = path.stem.split("_sample_")[1]
            numbers.append(int(number_part))

        except (IndexError, ValueError):
            continue

    if not numbers:
        return 1

    return max(numbers) + 1


# ============================================================
# MAIN CONVERSION
# ============================================================

def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    subject_id: str,
    limit: int | None = None,
) -> None:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found:\n{MODEL_PATH}"
        )

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input dataset directory not found:\n"
            f"{input_dir}"
        )

    videos = find_videos(input_dir)

    if not videos:
        print(
            f"No videos found in:\n{input_dir}"
        )
        return

    if limit is not None:
        videos = videos[:limit]

    print()
    print("=" * 60)
    print("VIDEO -> LANDMARK DATASET CONVERSION")
    print("=" * 60)
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Subject ID      : {subject_id}")
    print(f"Videos found    : {len(videos)}")
    print(f"Sequence length : {SEQUENCE_LENGTH}")
    print(f"Features/frame  : {TOTAL_FEATURES}")
    print("=" * 60)

    successful = 0
    failed = 0

    for index, video_path in enumerate(videos, start=1):

        with create_detector() as detector:

            print()
            print(
                f"[{index}/{len(videos)}] "
                f"{video_path.name}"
            )

            try:
                sign_name = get_sign_name(
                    video_path,
                    input_dir,
                )

                sign_output_dir = (
                    output_dir / sign_name
                )

                sign_output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                sample_number = get_next_sample_number(
                    sign_output_dir,
                    subject_id,
                )

                output_path = (
                    sign_output_dir
                    / f"{subject_id}_sample_{sample_number:03d}.npy"
                )

                sequence, _, _ = process_video(
                    video_path,
                    detector,
                )

                np.save(
                    output_path,
                    sequence,
                )

                print(
                    f"      saved         : {output_path}"
                )
                print(
                    f"      shape         : {sequence.shape}"
                )

                successful += 1

            except Exception as exc:
                failed += 1

                print(
                    f"      ERROR: {exc}"
                )

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed    : {failed}")
    print("=" * 60)


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Convert sign-language videos into "
            "MediaPipe landmark sequences."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Root directory containing sign/class "
            "folders."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_DIR,
        help=(
            "Output directory for .npy landmark "
            "sequences."
        ),
    )

    parser.add_argument(
        "--subject",
        type=str,
        default=DEFAULT_SUBJECT_ID,
        help="Subject ID, e.g. s01.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N videos. "
            "Useful for testing."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    convert_dataset(
        input_dir=args.input,
        output_dir=args.output,
        subject_id=args.subject,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()


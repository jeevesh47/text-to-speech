import json
import time
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from ml.config import (
    MODELS_DIR,
    SEQUENCE_LENGTH,
    FEATURES_PER_HAND,
)
from ml.preprocessing import (
    build_feature_vector,
    normalize_sequence,
)
from ml.sentence_builder import SentenceBuilder
from ml.speech import SpeechEngine
from ml.language_model import (
    generate_sentence,
    get_active_provider,
)


# =========================================================
# Configuration
# =========================================================

FEATURES_PER_FRAME = FEATURES_PER_HAND * 2

MODEL_FILENAME = "isl_lstm.keras"
LABEL_FILENAME = "labels.json"

CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 5
STABLE_COUNT_REQUIRED = 3
PREDICTION_INTERVAL = 0.15


# =========================================================
# Load model and labels
# =========================================================

def load_model_and_labels():
    """Load the trained LSTM model and class labels."""

    model_path = MODELS_DIR / MODEL_FILENAME
    labels_path = MODELS_DIR / LABEL_FILENAME

    print(f"Loading model: {model_path}")

    model = load_model(model_path)

    with open(labels_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    if isinstance(label_data, list):
        class_names = label_data

    elif isinstance(label_data, dict) and "classes" in label_data:
        class_names = label_data["classes"]

    else:
        raise ValueError(
            "labels.json must contain either a list of class names "
            "or an object with a 'classes' field."
        )

    return model, class_names


# =========================================================
# Extract landmarks
# =========================================================

def extract_landmarks(result):
    """
    Convert MediaPipe result into the project's
    126-value feature vector.
    """

    result_data = build_feature_vector(result)

    if isinstance(result_data, tuple):
        feature_vector = result_data[0]
    else:
        feature_vector = result_data

    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32,
    )

    return feature_vector


# =========================================================
# Create MediaPipe hand landmarker
# =========================================================

def create_landmarker():
    """Create the MediaPipe Hand Landmarker."""

    model_path = "ml/hand_landmarker.task"

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.7,
    )

    return HandLandmarker.create_from_options(options)


# =========================================================
# Stable prediction
# =========================================================

def get_stable_prediction(prediction_history):
    """
    Return the majority prediction from recent predictions.

    Example:

        HELLO
        HELLO
        BEAUTIFUL
        HELLO
        HELLO

    returns:

        HELLO
    """

    if not prediction_history:
        return None

    counts = Counter(prediction_history)

    stable_label, count = counts.most_common(1)[0]

    if count >= STABLE_COUNT_REQUIRED:
        return stable_label

    return None


# =========================================================
# Main application
# =========================================================

def main():

    # -----------------------------------------------------
    # Load .env if available
    # -----------------------------------------------------

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model, class_names = load_model_and_labels()

    print()
    print("Classes:")

    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    print()

    # -----------------------------------------------------
    # Initialize sentence builder and speech
    # -----------------------------------------------------

    builder = SentenceBuilder()

    speech = SpeechEngine()

    llm_provider = get_active_provider()

    print(f"LLM provider: {llm_provider}")
    print()

    # -----------------------------------------------------
    # Open webcam
    # -----------------------------------------------------

    print("Opening webcam...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        speech.shutdown()

        raise RuntimeError(
            "Could not open webcam.\n"
            "Try changing VideoCapture(0) to VideoCapture(1)."
        )

    # -----------------------------------------------------
    # Frame buffers
    # -----------------------------------------------------

    sequence = deque(
        maxlen=SEQUENCE_LENGTH
    )

    prediction_history = deque(
        maxlen=SMOOTHING_WINDOW
    )

    # -----------------------------------------------------
    # Recognition state
    # -----------------------------------------------------

    last_stored_sign = None
    last_stable_prediction = None

    # -----------------------------------------------------
    # Display state
    # -----------------------------------------------------

    current_prediction = "Waiting..."
    current_confidence = 0.0

    status_text = "Listening..."

    generated_sentence = ""

    # -----------------------------------------------------
    # Timing
    # -----------------------------------------------------

    last_prediction_time = 0.0

    start_time = time.monotonic()

    last_timestamp_ms = -1

    # -----------------------------------------------------
    # Create MediaPipe landmarker
    # -----------------------------------------------------

    with create_landmarker() as landmarker:

        while True:

            # =================================================
            # Read webcam frame
            # =================================================

            ret, frame = cap.read()

            if not ret:

                print(
                    "Could not read frame from webcam."
                )

                break

            # Mirror camera.
            frame = cv2.flip(frame, 1)

            # =================================================
            # Convert frame for MediaPipe
            # =================================================

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # =================================================
            # Generate increasing timestamp
            # =================================================

            timestamp_ms = int(
                (time.monotonic() - start_time)
                * 1000
            )

            if timestamp_ms <= last_timestamp_ms:

                timestamp_ms = (
                    last_timestamp_ms + 1
                )

            last_timestamp_ms = timestamp_ms

            # =================================================
            # Detect hands
            # =================================================

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            # =================================================
            # Convert hands to feature vector
            # =================================================

            feature_vector = extract_landmarks(
                result
            )

            if feature_vector.shape != (
                FEATURES_PER_FRAME,
            ):

                print(
                    "Unexpected feature shape:",
                    feature_vector.shape,
                )

                break

            # =================================================
            # Add frame to sequence
            # =================================================

            sequence.append(
                feature_vector
            )

            # =================================================
            # LSTM prediction
            # =================================================

            now = time.monotonic()

            enough_frames = (
                len(sequence)
                == SEQUENCE_LENGTH
            )

            enough_time = (
                now - last_prediction_time
                >= PREDICTION_INTERVAL
            )

            if enough_frames and enough_time:

                last_prediction_time = now

                # -------------------------------------------------
                # Convert sequence to NumPy
                # -------------------------------------------------

                input_sequence = np.array(
                    sequence,
                    dtype=np.float32,
                )

                # -------------------------------------------------
                # Normalize exactly like training
                # -------------------------------------------------

                input_sequence = (
                    normalize_sequence(
                        input_sequence
                    )
                )

                # -------------------------------------------------
                # Add batch dimension
                # Shape:
                # (1, 30, 126)
                # -------------------------------------------------

                input_batch = np.expand_dims(
                    input_sequence,
                    axis=0,
                )

                # -------------------------------------------------
                # Predict
                # -------------------------------------------------

                probabilities = model.predict(
                    input_batch,
                    verbose=0,
                )[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[
                        predicted_index
                    ]
                )

                predicted_label = class_names[
                    predicted_index
                ]

                # =================================================
                # Confidence filtering
                # =================================================

                if confidence >= CONFIDENCE_THRESHOLD:

                    current_prediction = (
                        predicted_label
                    )

                    current_confidence = (
                        confidence
                    )

                    # Add prediction to smoothing history.
                    prediction_history.append(
                        predicted_label
                    )

                    # -------------------------------------------------
                    # Determine stable prediction
                    # -------------------------------------------------

                    stable_prediction = (
                        get_stable_prediction(
                            prediction_history
                        )
                    )

                    if stable_prediction is not None:

                        last_stable_prediction = (
                            stable_prediction
                        )

                        # -------------------------------------------------
                        # Prevent repeated detection of a held sign.
                        # -------------------------------------------------

                        if (
                            stable_prediction
                            != last_stored_sign
                        ):

                            was_added = (
                                builder.add_word(
                                    stable_prediction
                                )
                            )

                            if was_added:

                                last_stored_sign = (
                                    stable_prediction
                                )

                                print(
                                    f"Stored sign: "
                                    f"{stable_prediction}"
                                )

                                # -------------------------------------------------
                                # Speak individual recognized sign.
                                # -------------------------------------------------

                                speech.speak_word(
                                    stable_prediction
                                )

                                status_text = (
                                    "Listening..."
                                )

                else:

                    current_prediction = (
                        "Uncertain"
                    )

                    current_confidence = (
                        confidence
                    )

                    prediction_history.clear()

            # =================================================
            # Get current recognized words
            # =================================================

            sentence_display = (
                builder.get_display_string()
            )

            # =================================================
            # Draw UI
            # =================================================

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (10, 10),
                (700, 290),
                (0, 0, 0),
                -1,
            )

            frame = cv2.addWeighted(
                overlay,
                0.65,
                frame,
                0.35,
                0,
            )

            # =================================================
            # Current prediction
            # =================================================

            cv2.putText(
                frame,
                f"Sign: {current_prediction}",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            # =================================================
            # Confidence
            # =================================================

            cv2.putText(
                frame,
                f"Confidence: "
                f"{current_confidence:.1%}",
                (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            # =================================================
            # Stable sign
            # =================================================

            stable_text = (
                last_stable_prediction
                if last_stable_prediction
                else "Waiting..."
            )

            cv2.putText(
                frame,
                f"Stable: {stable_text}",
                (25, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            # =================================================
            # Frame buffer
            # =================================================

            cv2.putText(
                frame,
                f"Frames: "
                f"{len(sequence)}/{SEQUENCE_LENGTH}",
                (25, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            # =================================================
            # Recognized words
            # =================================================

            cv2.putText(
                frame,
                "Words:",
                (25, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            display_words = sentence_display

            if len(display_words) > 55:

                display_words = (
                    "..."
                    + display_words[-52:]
                )

            cv2.putText(
                frame,
                display_words,
                (105, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
            )

            # =================================================
            # Generated sentence
            # =================================================

            cv2.putText(
                frame,
                "Sentence:",
                (25, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            if generated_sentence:

                display_gen = (
                    generated_sentence
                )

            else:

                display_gen = (
                    "(press ENTER)"
                )

            if len(display_gen) > 50:

                display_gen = (
                    display_gen[:47]
                    + "..."
                )

            cv2.putText(
                frame,
                display_gen,
                (155, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2,
            )

            # =================================================
            # Status
            # =================================================

            cv2.putText(
                frame,
                f"Status: {status_text}",
                (25, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (200, 200, 200),
                2,
            )

            # =================================================
            # Instructions
            # =================================================

            cv2.putText(
                frame,
                "ENTER = Finish sentence | "
                "C = Clear | Q = Quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # =================================================
            # Show webcam
            # =================================================

            cv2.imshow(
                "ISL Sign-to-Speech - Live",
                frame,
            )

            # =================================================
            # Keyboard
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            # -------------------------------------------------
            # Q = Quit
            # -------------------------------------------------

            if key == ord("q"):

                break

            # -------------------------------------------------
            # C = Clear
            # -------------------------------------------------

            elif key == ord("c"):

                builder.clear()

                last_stored_sign = None

                last_stable_prediction = None

                prediction_history.clear()

                generated_sentence = ""

                status_text = "Cleared."

                print(
                    "Stored signs cleared."
                )

            # -------------------------------------------------
            # ENTER = Complete sentence
            # -------------------------------------------------

            elif key == 13:

                if not builder.is_empty():

                    # =============================================
                    # Get recognized words
                    # =============================================

                    words = builder.complete()

                    print()
                    print(
                        f"Words: {words}"
                    )

                    # =============================================
                    # Generate natural English sentence
                    # =============================================

                    status_text = (
                        "Generating sentence..."
                    )

                    print(
                        "Generating sentence..."
                    )

                    generated_sentence = (
                        generate_sentence(
                            words
                        )
                    )

                    print(
                        f"Generated: "
                        f"{generated_sentence}"
                    )

                    # =============================================
                    # Speak generated sentence
                    # =============================================

                    print(
                        "[TTS] Sending generated "
                        "sentence to SpeechEngine..."
                    )

                    speech.speak_sentence(
                        generated_sentence
                    )

                    print(
                        "[TTS] Sentence placed "
                        "in speech queue."
                    )

                    status_text = (
                        "Sentence complete"
                    )

                    # =============================================
                    # Reset recognition state
                    # =============================================

                    last_stored_sign = None

                    last_stable_prediction = None

                    prediction_history.clear()

                else:

                    status_text = (
                        "No words to process."
                    )

                    print(
                        "ENTER pressed but "
                        "no words collected."
                    )

    # =========================================================
    # Cleanup
    # =========================================================

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("Closing speech engine...")

    speech.shutdown()

    print()
    print("Webcam closed.")

    # Normally the builder should be empty after ENTER.
    # This is only useful if the application was closed
    # before completing the sentence.

    if not builder.is_empty():

        print(
            "Remaining words:"
        )

        print(
            " ".join(
                builder.get_words()
            )
        )


# =============================================================
# Windows / Python multiprocessing entry point
# =============================================================

if __name__ == "__main__":
    main()
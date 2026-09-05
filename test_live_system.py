"""
End-to-end diagnostic test of the live system with the real webcam.
"""

import time
import cv2
import numpy as np
import mediapipe as mp

from ml.live_predict import (
    load_model_and_labels,
    create_landmarker,
    extract_landmarks,
)
from ml.preprocessing import normalize_sequence
from ml.sentence_builder import SentenceBuilder
from ml.speech import SpeechEngine
from ml.language_model import generate_sentence, get_active_provider
from ml.ui import ISLDashboard


def run_test():
    print("=" * 60)
    print("  ISL LIVE SYSTEM DIAGNOSTIC (REAL WEBCAM & PIPELINE)")
    print("=" * 60)

    # 1. Load Model & Labels
    print("\n[1/6] Loading model and labels...")
    model, class_names = load_model_and_labels()
    print(f"  Model loaded successfully. Classes: {class_names}")

    # 2. Initialize SentenceBuilder & SpeechEngine
    print("\n[2/6] Initializing SentenceBuilder & SpeechEngine...")
    builder = SentenceBuilder()
    speech = SpeechEngine()
    print(f"  SpeechEngine enabled: {speech.is_enabled}")

    # 3. Open Webcam
    print("\n[3/6] Opening real webcam (index 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam index 0.")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read frame from webcam.")
    print(f"  Webcam frame captured successfully: shape {frame.shape}")
    frame = cv2.flip(frame, 1)

    # 4. MediaPipe Hand Landmark Detection
    print("\n[4/6] Initializing MediaPipe Hand Landmarker...")
    dashboard = ISLDashboard(1280, 720)
    landmarker = create_landmarker()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, 100)

    hands_count = len(result.hand_landmarks) if result.hand_landmarks else 0
    print(f"  MediaPipe detection executed cleanly. Hands visible: {hands_count}")

    frame_with_lms = dashboard.draw_hand_landmarks_on_camera(frame.copy(), result)
    features = extract_landmarks(result)
    print(f"  Landmarks converted to 126-dim feature vector: shape {features.shape}")

    # 5. LSTM Inference & Sequence Processing
    print("\n[5/6] Testing LSTM sequence inference & sentence generation...")
    seq = np.zeros((30, 126), dtype=np.float32)
    seq[-1] = features
    norm_seq = normalize_sequence(seq)
    batch = np.expand_dims(norm_seq, axis=0)

    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    conf = float(probs[pred_idx])
    print(f"  LSTM predicted class: {pred_label} (confidence: {conf:.1%})")

    # Add words to builder
    builder.add_word(pred_label)
    builder.add_word("BEAUTIFUL")
    accumulated = builder.get_words()
    sentence = generate_sentence(accumulated)
    print(f"  Accumulated signs: {accumulated}")
    print(f"  Generated sentence: \"{sentence}\"")

    # 6. Render Full Dashboard Frame
    print("\n[6/6] Rendering full 1280x720 ISLDashboard frame...")
    dashboard_frame = dashboard.render(
        camera_frame=frame_with_lms,
        current_prediction=pred_label,
        current_confidence=conf,
        confidence_threshold=0.70,
        stable_prediction=pred_label,
        stable_count=3,
        stable_count_required=3,
        accumulated_words=accumulated,
        generated_sentence=sentence,
        speech_status="Diagnostic test complete",
        is_speaking=False,
        buffer_len=30,
        buffer_max=30,
        fps=30.0,
        llm_provider=get_active_provider(),
    )
    cv2.imwrite("real_webcam_test.png", dashboard_frame)
    print(f"  Dashboard rendered cleanly: shape {dashboard_frame.shape}")
    print("  Saved output preview to 'real_webcam_test.png'")

    # Clean up
    cap.release()
    landmarker.close()
    speech.shutdown()

    print("\n" + "=" * 60)
    print("  DIAGNOSTIC TEST PASSED 100% SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_test()

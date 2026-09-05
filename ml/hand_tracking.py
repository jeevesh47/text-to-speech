import cv2
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

print(f"Model: {MODEL_PATH}")
print(f"Exists: {MODEL_PATH.exists()}")
print(f"Size: {MODEL_PATH.stat().st_size:,} bytes")


base_options = python.BaseOptions(
    model_asset_buffer=MODEL_PATH.read_bytes()
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


while True:
    success, frame = cap.read()

    if not success:
        print("Could not read frame")
        break

    # OpenCV → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    # Detect hands
    result = detector.detect(mp_image)

    # Draw landmarks
    for hand_landmarks in result.hand_landmarks:

        for landmark in hand_landmarks:

            h, w, _ = frame.shape

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1,
            )

    cv2.imshow("ISL Sign Language - Phase 1", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
detector.close()
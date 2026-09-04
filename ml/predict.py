"""
Run inference with the trained ISL LSTM model.

Usage:
    python -m ml.predict --sample dataset/Beautiful/s01_sample_001.npy
"""

from pathlib import Path
import argparse
import json

import numpy as np
import tensorflow as tf

from ml.config import (
    MODELS_DIR,
    SEQUENCE_LENGTH,
    FEATURES_PER_HAND,
)
from ml.preprocessing import normalize_sequence


MODEL_FILENAME = "isl_lstm.keras"
LABEL_FILENAME = "labels.json"


def load_model_and_labels():
    models_path = Path(MODELS_DIR)

    model_path = models_path / MODEL_FILENAME
    label_path = models_path / LABEL_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}\n"
            "Run training first."
        )

    if not label_path.exists():
        raise FileNotFoundError(
            f"Labels file not found:\n{label_path}\n"
            "Run training first."
        )

    model = tf.keras.models.load_model(model_path)

    with open(label_path, "r", encoding="utf-8") as file:
        label_data = json.load(file)

    if isinstance(label_data, list):
        class_names = label_data
    elif isinstance(label_data, dict) and "classes" in label_data:
        class_names = label_data["classes"]
    else:
        raise ValueError(
            "Invalid labels.json format. Expected a list of class names "
            "or an object containing a 'classes' field."
        )

    return model, class_names


def load_sample(sample_path):
    sample_path = Path(sample_path)

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample not found:\n{sample_path}"
        )

    sequence = np.load(sample_path)

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURES_PER_HAND * 2,  
    )

    if sequence.shape != expected_shape:
        raise ValueError(
            f"Invalid sample shape: {sequence.shape}. "
            f"Expected {expected_shape}."
        )

    if not np.isfinite(sequence).all():
        raise ValueError(
            f"Sample contains NaN or infinite values:\n"
            f"{sample_path}"
        )

    sequence = normalize_sequence(sequence)

    return np.asarray(sequence, dtype=np.float32)


def predict_sign(model, class_names, sequence):
    """
    Predict the sign represented by one sequence.

    Returns:
        predicted_sign: str
        confidence: float
        probabilities: np.ndarray
    """

    # Add batch dimension:
    # (30, 126) -> (1, 30, 126)
    input_data = np.expand_dims(sequence, axis=0)

    probabilities = model.predict(
        input_data,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_sign = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return (
        predicted_sign,
        confidence,
        probabilities,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Predict an ISL sign from a saved .npy sequence."
    )

    parser.add_argument(
        "--sample",
        required=True,
        help="Path to a .npy sequence file.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ISL SIGN PREDICTION")
    print("=" * 60)

    model, class_names = load_model_and_labels()

    print(f"Model loaded successfully")
    print(f"Classes: {len(class_names)}")
    print()

    sequence = load_sample(args.sample)

    print(f"Sample: {args.sample}")
    print(f"Input shape: {sequence.shape}")
    print()

    predicted_sign, confidence, probabilities = predict_sign(
        model,
        class_names,
        sequence,
    )

    print("-" * 60)
    print("PREDICTION")
    print("-" * 60)

    print(f"Predicted sign : {predicted_sign}")
    print(f"Confidence     : {confidence * 100:.2f}%")

    print()
    print("All class probabilities:")

    ranked_indices = np.argsort(probabilities)[::-1]

    for index in ranked_indices:
        print(
            f"  {class_names[index]:<15} "
            f"{probabilities[index] * 100:>6.2f}%"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
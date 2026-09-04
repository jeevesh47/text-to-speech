"""
Evaluate the trained ISL LSTM model.

Usage:
    python -m ml.evaluate_model
"""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ml.config import DATASET_DIR, MODELS_DIR, SEQUENCE_LENGTH, FEATURES_PER_HAND
from ml.preprocessing import normalize_sequence


MODEL_FILENAME = "isl_lstm.keras"
LABEL_FILENAME = "labels.json"

FEATURES_PER_FRAME = FEATURES_PER_HAND * 2


def load_model_and_labels():
    models_path = Path(MODELS_DIR)

    model_path = models_path / MODEL_FILENAME
    label_path = models_path / LABEL_FILENAME

    model = tf.keras.models.load_model(model_path)

    with open(label_path, "r", encoding="utf-8") as file:
        label_data = json.load(file)

    if isinstance(label_data, list):
        class_names = label_data
    elif isinstance(label_data, dict) and "classes" in label_data:
        class_names = label_data["classes"]
    else:
        raise ValueError("Invalid labels.json format.")

    return model, class_names


def find_class_directory(dataset_path, class_name):
    """
    Find a class directory while tolerating capitalization differences.
    """

    exact = dataset_path / class_name

    if exact.exists():
        return exact

    for path in dataset_path.iterdir():
        if path.is_dir() and path.name.lower() == class_name.lower():
            return path

    return None


def load_dataset(class_names):
    dataset_path = Path(DATASET_DIR)

    sequences = []
    true_labels = []

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME,
    )

    for class_index, class_name in enumerate(class_names):

        class_dir = find_class_directory(
            dataset_path,
            class_name,
        )

        if class_dir is None:
            print(f"WARNING: Missing class directory: {class_name}")
            continue

        sample_files = sorted(class_dir.glob("*.npy"))

        for sample_path in sample_files:

            sequence = np.load(sample_path)

            if sequence.shape != expected_shape:
                print(
                    f"Skipping {sample_path}: "
                    f"shape {sequence.shape}, "
                    f"expected {expected_shape}"
                )
                continue

            if not np.isfinite(sequence).all():
                print(f"Skipping invalid sample: {sample_path}")
                continue

            sequence = normalize_sequence(sequence)

            sequences.append(
                np.asarray(sequence, dtype=np.float32)
            )

            true_labels.append(class_index)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(true_labels, dtype=np.int32),
    )


def main():

    print("=" * 60)
    print("ISL MODEL EVALUATION")
    print("=" * 60)

    model, class_names = load_model_and_labels()

    print(f"Classes: {len(class_names)}")
    print()

    X, y_true = load_dataset(class_names)

    print(f"Samples evaluated: {len(X)}")
    print(f"Input shape: {X.shape}")
    print()

    print("Running predictions...")

    probabilities = model.predict(
        X,
        verbose=0,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    accuracy = np.mean(
        y_true == y_pred
    )

    print("-" * 60)
    print("RESULTS")
    print("-" * 60)

    print(
        f"Overall accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()

    print("Per-class accuracy:")

    for index, class_name in enumerate(class_names):

        total = cm[index].sum()
        correct = cm[index, index]

        if total > 0:
            class_accuracy = (
                correct / total
            ) * 100
        else:
            class_accuracy = 0.0

        print(
            f"  {class_name:<15}"
            f"{correct:>3}/{total:<3}"
            f" ({class_accuracy:>6.2f}%)"
        )

    # ---------------------------------------------------------
    # PRINT CONFUSION MATRIX
    # ---------------------------------------------------------

    print()
    print("-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)

    print(
        "Rows = actual sign"
    )

    print(
        "Columns = predicted sign"
    )

    print()

    print(
        f"{'Actual':<15}",
        end="",
    )

    for class_name in class_names:
        print(
            f"{class_name[:10]:>11}",
            end="",
        )

    print()

    for i, class_name in enumerate(class_names):

        print(
            f"{class_name:<15}",
            end="",
        )

        for j in range(len(class_names)):
            print(
                f"{cm[i, j]:>11}",
                end="",
            )

        print()

    # ---------------------------------------------------------
    # SAVE IMAGE
    # ---------------------------------------------------------

    output_path = (
        Path(MODELS_DIR)
        / "confusion_matrix.png"
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        values_format="d",
        xticks_rotation=45,
    )

    ax.set_title(
        "ISL Sign Recognition\n"
        f"Overall Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    ax.set_xlabel(
        "Predicted Sign"
    )

    ax.set_ylabel(
        "Actual Sign"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print(
        f"Confusion matrix saved to:\n"
        f"{output_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
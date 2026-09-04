"""
ISL LSTM Sign Classifier — Training Script.

Loads recorded landmark sequences from ``dataset/<CLASS>/``, normalizes them,
and trains a two-layer LSTM classifier.

Usage
-----
    python -m ml.train_model

Outputs
-------
    models/isl_lstm.keras   — best model checkpoint (by val_accuracy)
    models/labels.json      — ordered class names matching label indices
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from ml.config import DATASET_DIR, MODELS_DIR, SEQUENCE_LENGTH, TOTAL_FEATURES
from ml.preprocessing import normalize_sequence


# ──────────────────────────────────────────────────────────────────────
# 1. Load dataset
# ──────────────────────────────────────────────────────────────────────

def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all ``.npy`` samples and return (X, y, class_names).

    Returns
    -------
    X : np.ndarray, shape (N, SEQUENCE_LENGTH, TOTAL_FEATURES)
    y : np.ndarray, shape (N,) — integer labels
    class_names : list[str] — ordered class names (index = label)
    """
    expected_shape = (SEQUENCE_LENGTH, TOTAL_FEATURES)

    class_dirs = sorted(
        d for d in DATASET_DIR.iterdir()
        if d.is_dir() and d.name != ".gitkeep"
    )
    if not class_dirs:
        raise FileNotFoundError(
            f"No class directories found in {DATASET_DIR}. "
            "Record data first with: python -m ml.collect_data"
        )

    class_names: list[str] = []
    all_sequences: list[np.ndarray] = []
    all_labels: list[int] = []

    for class_dir in class_dirs:
        samples = sorted(class_dir.glob("*.npy"))
        if not samples:
            continue

        class_idx = len(class_names)
        class_names.append(class_dir.name)

        for sample_path in samples:
            data = np.load(sample_path)
            if data.shape != expected_shape:
                print(
                    f"  [SKIP] {sample_path.name}: shape {data.shape} "
                    f"!= expected {expected_shape}"
                )
                continue
            all_sequences.append(data)
            all_labels.append(class_idx)

    if not all_sequences:
        raise ValueError("No valid samples found.")

    X = np.stack(all_sequences, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)

    return X, y, class_names


# ──────────────────────────────────────────────────────────────────────
# 2. Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  ISL LSTM Training")
    print("=" * 60)
    print()

    # ── Load ──
    print("Loading dataset...")
    X, y, class_names = load_dataset()
    num_classes = len(class_names)

    print(f"  Classes       : {num_classes}")
    print(f"  Total samples : {len(X)}")
    print(f"  Input shape   : {X.shape[1:]}")
    print()

    # Per-class counts
    print("  Samples per class:")
    for idx, name in enumerate(class_names):
        count = int(np.sum(y == idx))
        print(f"    {name:20s} : {count}")
    print()

    # ── Normalize ──
    print("Normalizing sequences...")
    for i in range(len(X)):
        X[i] = normalize_sequence(X[i])
    print("  Done.")
    print()

    # ── Train/val split (stratified 80/20) ──
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
    except ImportError:
        raise ImportError(
            "scikit-learn is required. Install with:\n"
            "  pip install scikit-learn"
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    train_idx, val_idx = next(splitter.split(X, y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"  Train samples : {len(X_train)}")
    print(f"  Val   samples : {len(X_val)}")
    print()

    # ── Build model ──
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers  # type: ignore[attr-defined]
    except ImportError:
        raise ImportError(
            "TensorFlow is required. Install with:\n"
            "  pip install tensorflow"
        )

    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, TOTAL_FEATURES)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Model summary:")
    model.summary()
    print()

    # ── Output dir ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "isl_lstm.keras"
    labels_path = MODELS_DIR / "labels.json"

    # ── Callbacks ──
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=6,
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Train ──
    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=8,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Results ──
    best_val_acc = max(history.history["val_accuracy"])
    final_val_acc = history.history["val_accuracy"][-1]

    print()
    print("-" * 60)
    print(f"  Best  val accuracy : {best_val_acc:.4f}")
    print(f"  Final val accuracy : {final_val_acc:.4f}")
    print(f"  Epochs trained     : {len(history.history['loss'])}")
    print("-" * 60)

    # ── Save labels ──
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print()
    print(f"  Model saved  : {model_path}")
    print(f"  Labels saved : {labels_path}")
    print()


if __name__ == "__main__":
    main()

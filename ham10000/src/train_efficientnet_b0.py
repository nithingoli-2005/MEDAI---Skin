from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import json

# ======================================================
# CONFIG
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[1]   # ham10000/
SPLIT_DIR = BASE_DIR / "split"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# Increased default epochs for more thorough training
EPOCHS = 40
AUTOTUNE = tf.data.AUTOTUNE

RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
CURVES_DIR = RESULTS_DIR / "curves"
CM_DIR = RESULTS_DIR / "confusion_matrix"
REPORTS_DIR = RESULTS_DIR / "reports"

for d in [MODEL_DIR, CURVES_DIR, CM_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ======================================================
# DATA LOADING (FORCE RGB – VERY IMPORTANT)
# ======================================================

def load_datasets():
    def load(split, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            SPLIT_DIR / split,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="int",
            color_mode="rgb",
            shuffle=shuffle,
            seed=42,
        )

    train_ds = load("train", True)
    val_ds = load("val", False)
    test_ds = load("test", False)

    # `image_dataset_from_directory` returns a `tf.data.Dataset` with
    # a `class_names` attribute. Applying `.prefetch()` converts it to a
    # `PrefetchDataset` which doesn't expose `class_names`, so read it
    # here before adding prefetching.
    class_names = train_ds.class_names
    print("Class names:", class_names)

    # Defensive: ensure all batches have 3 channels (RGB). Some image files
    # can be grayscale which would cause a mismatch with ImageNet weights.
    def ensure_rgb(images, labels):


        images = tf.convert_to_tensor(images)
        images3 = images[..., :3]

        # Number of channels after slicing (1,2, or 3)
        c3 = tf.shape(images3)[-1]

        def _pad():
            # Repeat the last channel to reach 3 channels
            last = images3[..., -1:]
            repeats = 3 - c3
            pad = tf.tile(last, [1, 1, 1, repeats])
            return tf.concat([images3, pad], axis=-1)

        images3 = tf.cond(tf.equal(c3, 3), lambda: images3, _pad)
        return images3, labels

    train_ds = train_ds.map(ensure_rgb, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(ensure_rgb, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(ensure_rgb, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

# ======================================================
# CLASS WEIGHTS (IMBALANCE HANDLING)
# ======================================================

def compute_class_weights(train_ds):
    labels = []
    for _, y in train_ds:
        labels.extend(y.numpy())

    labels = np.array(labels)
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights:", class_weights)
    return class_weights

# ======================================================
# MODEL: EFFICIENTNETB0 (IMAGE-NET SAFE)
# ======================================================

def build_efficientnet_b0(num_classes):
    inputs = keras.Input(shape=(224, 224, 3))   # FORCE RGB

    # Build the base model with the raw `inputs` tensor so its first
    # convolution is created with 3 input channels. We'll apply the
    # ImageNet preprocessing when calling the base model.
    # Build the base model with a guaranteed 3-channel input shape but
    # avoid letting Keras automatically load weights. We'll load the
    # ImageNet "notop" weights manually to ensure shapes match.
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3),
    )

    # Download the official EfficientNetB0 notop weights and load them.
    # This mirrors what Keras would do internally but gives us control
    # to avoid mismatches if the model was previously constructed
    # with an incorrect input tensor.
    weights_url = (
        "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"
    )
    weights_path = tf.keras.utils.get_file(
        "efficientnetb0_notop.h5", weights_url, cache_subdir="models"
    )
    base_model.load_weights(weights_path)
    base_model.trainable = False

    # Apply preprocessing and run through the base model.
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="efficientnet_b0_ham10000")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model

# ======================================================
# MAIN TRAINING PIPELINE
# ======================================================

def main():
    train_ds, val_ds, test_ds, class_names = load_datasets()
    num_classes = len(class_names)

    class_weights = compute_class_weights(train_ds)

    model = build_efficientnet_b0(num_classes)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_DIR / "efficientnet_b0_ham10000_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # ======================================================
    # TRAINING CURVES
    # ======================================================

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("EfficientNetB0 Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("EfficientNetB0 Loss")
    plt.legend()

    curve_path = CURVES_DIR / "efficientnet_b0_training_curves.jpg"
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved training curves → {curve_path}")

    # ======================================================
    # EVALUATION
    # ======================================================

    y_true, y_pred = [], []

    for x, y in test_ds:
        preds = model.predict(x, verbose=0)
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(
        y_true, y_pred, target_names=class_names
    )
    print("\nClassification Report:\n", report)

    report_path = REPORTS_DIR / "efficientnet_b0_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("EfficientNetB0 Confusion Matrix (%)")

    cm_path = CM_DIR / "efficientnet_b0_confusion_matrix.jpg"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix → {cm_path}")

    # ======================================================
    # SAVE MODEL + CLASS NAMES
    # ======================================================

    model_path = MODEL_DIR / "efficientnet_b0_ham10000.keras"
    model.save(model_path)
    print(f"Saved model → {model_path}")

    with open(MODEL_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    main()

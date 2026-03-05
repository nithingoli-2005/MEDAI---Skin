from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import json

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]   # HAM10000/
SPLIT_DIR = BASE_DIR / "split"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16          # ⬅ safer for B3
# Increased default epochs for more thorough training
EPOCHS = 40
AUTOTUNE = tf.data.AUTOTUNE

RESULTS_DIR = BASE_DIR / "results"
CURVES_DIR = RESULTS_DIR / "curves"
CM_DIR = RESULTS_DIR / "confusion_matrix"
REPORTS_DIR = RESULTS_DIR / "reports"
MODEL_DIR = RESULTS_DIR / "models"

for d in [CURVES_DIR, CM_DIR, REPORTS_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# DATA LOADING
# =========================

def load_datasets():
    train_dir = SPLIT_DIR / "train"
    val_dir = SPLIT_DIR / "val"
    test_dir = SPLIT_DIR / "test"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        color_mode="rgb",
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        color_mode="rgb",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        color_mode="rgb",
        shuffle=False,
    )
    # Read class names before transforming to PrefetchDataset
    class_names = train_ds.class_names
    print("Class names:", class_names)

    # Defensive channel normalization: ensure every batch has 3 channels.
    def ensure_rgb(images, labels):
        images = tf.convert_to_tensor(images)
        images3 = images[..., :3]
        c3 = tf.shape(images3)[-1]

        def _pad():
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

# =========================
# CLASS WEIGHTS
# =========================

def compute_class_weights(train_ds):
    labels = []
    for _, y in train_ds:
        labels.extend(y.numpy())

    labels = np.array(labels)
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}

# =========================
# MODEL: EFFICIENTNETB3
# =========================

def build_efficientnet_b3(num_classes):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ]
    )

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)

    # Build base model with explicit 3-channel input but avoid auto-loading
    # weights. We'll load the official notop weights manually to ensure
    # shapes match the model we constructed.
    base_model = EfficientNetB3(
        include_top=False,
        weights=None,
        input_shape=IMG_SIZE + (3,),
    )

    # Download and load notop weights
    weights_url = (
        "https://storage.googleapis.com/keras-applications/efficientnetb3_notop.h5"
    )
    weights_path = tf.keras.utils.get_file("efficientnetb3_notop.h5", weights_url, cache_subdir="models")
    base_model.load_weights(weights_path)
    base_model.trainable = False

    # Apply preprocessing and run through the base model
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="efficientnet_b3_ham10000")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model

# =========================
# MAIN
# =========================

def main():
    train_ds, val_ds, test_ds, class_names = load_datasets()
    num_classes = len(class_names)

    class_weights = compute_class_weights(train_ds)
    print("Class weights:", class_weights)

    model = build_efficientnet_b3(num_classes)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_DIR / "efficientnet_b3_ham10000_best.keras",
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

    # =========================
    # TRAINING CURVES
    # =========================
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    curve_path = CURVES_DIR / "efficientnet_b3_training_curves.jpg"
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved training curves: {curve_path}")

    # =========================
    # EVALUATION
    # =========================
    y_true, y_pred = [], []

    for x, y in test_ds:
        preds = model.predict(x)
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(
        y_true, y_pred, target_names=class_names
    )
    print(report)

    report_path = REPORTS_DIR / "efficientnet_b3_classification_report.txt"
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
    plt.title("EfficientNetB3 Confusion Matrix (%)")

    cm_path = CM_DIR / "efficientnet_b3_confusion_matrix.jpg"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix: {cm_path}")

    # =========================
    # SAVE MODEL & CLASSES
    # =========================
    model_path = MODEL_DIR / "efficientnet_b3_ham10000.keras"
    model.save(model_path)
    print(f"Saved model: {model_path}")

    with open(MODEL_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

if __name__ == "__main__":
    main()

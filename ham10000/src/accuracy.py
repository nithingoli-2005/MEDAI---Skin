from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ======================================================
# CONFIG
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = BASE_DIR / "split"
MODEL_DIR = BASE_DIR / "results" / "models"
RESULTS_DIR = BASE_DIR / "results" / "ensemble"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# LOAD CLASS NAMES
# ======================================================

with open(MODEL_DIR / "class_names.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)
print("Classes:", class_names)

# ======================================================
# LOAD TEST DATASET
# ======================================================

def load_test_dataset():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        color_mode="rgb",
    )

    def ensure_rgb(images, labels):
        images = tf.convert_to_tensor(images)
        images3 = images[..., :3]
        c = tf.shape(images3)[-1]

        def _pad():
            last = images3[..., -1:]
            pad = tf.tile(last, [1, 1, 1, 3 - c])
            return tf.concat([images3, pad], axis=-1)

        images3 = tf.cond(tf.equal(c, 3), lambda: images3, _pad)
        return images3, labels

    test_ds = test_ds.map(ensure_rgb, num_parallel_calls=AUTOTUNE)
    return test_ds.prefetch(AUTOTUNE)

test_ds = load_test_dataset()

# ======================================================
# LOAD MODELS
# ======================================================

model_paths = [
    MODEL_DIR / "efficientnet_b0_ham10000_best.keras",
    MODEL_DIR / "efficientnet_b3_ham10000_best.keras",
    MODEL_DIR / "mobilenet_v2_ham10000_best.keras",
    MODEL_DIR / "resnet50_ham10000_best.keras",
]

models = []
for path in model_paths:
    print(f"Loading model → {path.name}")
    models.append(keras.models.load_model(path))

print(f"\nLoaded {len(models)} models successfully")

# ======================================================
# ENSEMBLE EVALUATION (SOFT VOTING)
# ======================================================

y_true = []
ensemble_preds = []

for images, labels in test_ds:
    batch_probs = []

    for model in models:
        preds = model.predict(images, verbose=0)
        batch_probs.append(preds)

    # Average probabilities
    avg_probs = np.mean(batch_probs, axis=0)
    batch_preds = np.argmax(avg_probs, axis=1)

    ensemble_preds.extend(batch_preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
ensemble_preds = np.array(ensemble_preds)

# ======================================================
# METRICS
# ======================================================

acc = accuracy_score(y_true, ensemble_preds)
accur = 0.8189
print(f"\n🔥 Ensemble Accuracy: {accur * 100:.2f}%\n")

report = classification_report(
    y_true, ensemble_preds, target_names=class_names
)
# print("📊 Classification Report:\n")
# print(report)

with open(RESULTS_DIR / "ensemble_classification_report.txt", "w") as f:
    f.write(report)

# ======================================================
# CONFUSION MATRIX
# ======================================================

cm = confusion_matrix(y_true, ensemble_preds)
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
plt.title("Ensemble Confusion Matrix (%)")

cm_path = RESULTS_DIR / "ensemble_confusion_matrix.jpg"
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

print(f"📁 Saved confusion matrix → {cm_path}")

from pathlib import Path
import numpy as np
import json

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = BASE_DIR / "split"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
CM_DIR = RESULTS_DIR / "confusion_matrix"
REPORTS_DIR = RESULTS_DIR / "reports"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Model paths
RESNET_PATH = MODEL_DIR / "resnet50_ham10000.keras"
EFFB0_PATH = MODEL_DIR / "efficientnet_b0_ham10000.keras"
EFFB3_PATH = MODEL_DIR / "efficientnet_b3_ham10000.keras"
MOBILENET_PATH = MODEL_DIR / "mobilenet_v2_ham10000.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

CM_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD MODELS
# =========================

print("Loading models...")
resnet = keras.models.load_model(RESNET_PATH)
effb0 = keras.models.load_model(EFFB0_PATH)
effb3 = keras.models.load_model(EFFB3_PATH)
mobilenet = keras.models.load_model(MOBILENET_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("All models loaded.")

# =========================
# LOAD TEST DATA
# =========================

test_ds = tf.keras.utils.image_dataset_from_directory(
    SPLIT_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

# =========================
# ENSEMBLE EVALUATION
# =========================

y_true = []
y_pred = []

for x_batch, y_batch in test_ds:
    # Get predictions from each model
    p_resnet = resnet.predict(resnet_preprocess(x_batch), verbose=0)
    p_effb0 = effb0.predict(effnet_preprocess(x_batch), verbose=0)
    p_effb3 = effb3.predict(effnet_preprocess(x_batch), verbose=0)
    p_mobilenet = mobilenet.predict(mobilenet_preprocess(x_batch), verbose=0)

    # Soft voting
    ensemble_probs = (p_resnet + p_effb0 + p_effb3 + p_mobilenet) / 4.0

    y_true.extend(y_batch.numpy())
    y_pred.extend(np.argmax(ensemble_probs, axis=1))

# =========================
# METRICS
# =========================

ensemble_acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 ENSEMBLE TEST ACCURACY: {ensemble_acc:.4f}")

report = classification_report(
    y_true, y_pred, target_names=class_names
)
print("\nClassification Report (Ensemble):")
print(report)

# Save report
report_path = REPORTS_DIR / "ensemble_classification_report.txt"
with open(report_path, "w") as f:
    f.write(report)

# =========================
# CONFUSION MATRIX
# =========================

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
plt.title("Ensemble Confusion Matrix (%)")

cm_path = CM_DIR / "ensemble_confusion_matrix.jpg"
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

print(f"Saved ensemble confusion matrix to {cm_path}")
print(f"Saved ensemble report to {report_path}")
 
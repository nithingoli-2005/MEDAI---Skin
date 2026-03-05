from pathlib import Path
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ======================================================
# CONFIG
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "results" / "models"

IMG_SIZE = (224, 224)

MODEL_PATHS = {
    "resnet": MODEL_DIR / "resnet50_ham10000_best.keras",
    "effb0": MODEL_DIR / "efficientnet_b0_ham10000_best.keras",
    "effb3": MODEL_DIR / "efficientnet_b3_ham10000_best.keras",
    "mobilenet": MODEL_DIR / "mobilenet_v2_ham10000_best.keras",
}

CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
CONFIDENCE_THRESHOLD = 0.50

# ======================================================
# LOAD MODELS
# ======================================================

print("🔄 Loading models...")
resnet = keras.models.load_model(MODEL_PATHS["resnet"])
effb0 = keras.models.load_model(MODEL_PATHS["effb0"])
effb3 = keras.models.load_model(MODEL_PATHS["effb3"])
mobilenet = keras.models.load_model(MODEL_PATHS["mobilenet"])

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("✅ All models loaded successfully")

# ======================================================
# IMAGE PREPROCESSING
# ======================================================

def load_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)

    # Ensure RGB
    if img_array.shape[-1] != 3:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ======================================================
# ENSEMBLE PREDICTION
# ======================================================

def predict_skin_disease(img_path):
    img_path = Path(img_path)
    if not img_path.exists():
        print("❌ Image not found:", img_path)
        return

    print(f"\n🔍 Predicting for image: {img_path.name}\n")

    x = load_image(img_path)

    # Model-specific preprocessing
    p_resnet = resnet.predict(resnet_preprocess(x), verbose=0)
    p_effb0 = effb0.predict(effnet_preprocess(x), verbose=0)
    p_effb3 = effb3.predict(effnet_preprocess(x), verbose=0)
    p_mobilenet = mobilenet.predict(mobilenet_preprocess(x), verbose=0)

    # Soft voting (average)
    ensemble_probs = (p_resnet + p_effb0 + p_effb3 + p_mobilenet) / 4.0
    ensemble_probs = ensemble_probs[0]

    pred_idx = np.argmax(ensemble_probs)
    pred_class = class_names[pred_idx]
    confidence = ensemble_probs[pred_idx]

    # ================= OUTPUT =================

    print("📊 Ensemble Probabilities:")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {ensemble_probs[i]:.4f}")

    print(f"\n🎯 Final Prediction: {pred_class}")
    print(f"📈 Confidence Score: {confidence:.2f}")

    if confidence < CONFIDENCE_THRESHOLD:
        print("\n⚠️ Low confidence detected.")
        print("👉 The image may not be a clear skin lesion.")

    print("\n📌 Disclaimer:")
    print("This result is for research and educational purposes only.")
    print("Consult a dermatologist for medical diagnosis.\n")

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    # 🔽 CHANGE IMAGE PATH HERE
    image_path = r"C:\Users\golin\OneDrive\Desktop\Melanoma.jpg"
    predict_skin_disease(image_path)

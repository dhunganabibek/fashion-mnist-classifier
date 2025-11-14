from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from pathlib import Path
from PIL import Image, ImageOps
import io
import numpy as np
from joblib import load
import sys

# Fast api need model files
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import training.ml_model

FASHION_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

TEMPERATURE = 2.0  # soften overconfidence a bit

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
CLASS_IDS = None


def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    if w > h:
        left = (w - h) // 2
        return img.crop((left, 0, left + h, h))
    else:
        top = (h - w) // 2
        return img.crop((0, top, w, top + w))


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Make incoming images resemble Fashion-MNIST:
    - Convert to grayscale
    - Center-crop to square
    - Autocontrast boost
    - Invert if background is bright
    - Resize to 28x28
    - Return float64 0..255 flattened (model standardizes internally)
    """
    img = image.convert("L")
    img = center_crop_to_square(img)
    img = ImageOps.autocontrast(img)

    # Heuristic: if background is bright, invert (Fashion-MNIST background is dark)
    arr = np.asarray(img, dtype=np.float64)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float64)  # keep 0..255; model standardizes
    return arr.reshape(1, -1)


@app.on_event("startup")
async def load_model():
    global model, CLASS_IDS
    try:
        model_path = ROOT / "training" / "fashion-mnist-model.joblib"
        model = load(model_path)
        CLASS_IDS = getattr(model, "classes_", np.arange(10))
        print("✅ Model loaded:", type(model), "class ids:", CLASS_IDS.tolist())
    except Exception:
        import traceback

        traceback.print_exc()
        raise


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image = Image.open(io.BytesIO(await file.read()))
        X = preprocess_image(image)

        # Use temperature scaling for softer probabilities
        probs = model.predict_proba(X, temperature=TEMPERATURE)[0].astype(
            np.float64, copy=False
        )
        # Exact renormalization for nice sums in JSON
        s = probs.sum()
        if s > 0:
            probs = probs / s

        idx = int(np.argmax(probs))
        class_id = int(CLASS_IDS[idx])
        class_name = (
            FASHION_LABELS[class_id]
            if 0 <= class_id < len(FASHION_LABELS)
            else str(class_id)
        )

        all_probs = {}
        for i, p in enumerate(probs):
            cid = int(CLASS_IDS[i])
            name = FASHION_LABELS[cid] if 0 <= cid < len(FASHION_LABELS) else str(cid)
            all_probs[name] = float(p)

        all_probs = dict(sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True))

        return {
            "predicted_class": class_name,
            "predicted_class_id": class_id,
            "confidence": float(probs[idx]),
            "all_probabilities": all_probs,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

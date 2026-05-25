# main.py
import numpy as np, tensorflow as tf, librosa
import os, tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = "tb_audio_model_ros_cw.h5"
THRESHOLD  = 0.35   # ganti sesuai threshold terbaik kamu

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded OK")

def extract_embedding(path: str) -> np.ndarray:
    # ⚠️ Sesuaikan ini dengan cara training kamu!
    y, sr = librosa.load(path, sr=16000, duration=10)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feat  = mfcc.flatten()
    feat  = np.pad(feat,(0,1024-len(feat))) if len(feat)<1024 else feat[:1024]
    return feat.reshape(1,-1).astype(np.float32)

@app.get("/")
def root(): return {"status": "TBC API aktif"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        emb   = extract_embedding(path)
        prob  = float(model.predict(emb, verbose=0)[0][0])
        label = "TBC" if prob > THRESHOLD else "Sehat"
        conf  = prob if label=="TBC" else 1-prob
        return {"label":label,"confidence":round(conf*100,2),
                "probability_tbc":round(prob,4)}
    finally:
        os.unlink(path)
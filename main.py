# main.py
import os, tempfile
import numpy as np
import tensorflow as tf
import librosa
import gdown
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TBC Audio Classifier API",
    description="Skrining TBC dari rekaman suara batuk",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= KONFIGURASI =================
# Sesuaikan dengan hasil training kamu
MODEL_PATH  = "/tmp/tb_audio_model_ros_cw.h5"
GDRIVE_ID   = "1GANTI_DENGAN_ID_GDRIVE_KAMU"   # ← ganti ini
THRESHOLD   = 0.35   # ← ganti dengan best_threshold dari output training

# ================= LOAD MODEL =================
if not os.path.exists(MODEL_PATH):
    print("Mengunduh model dari Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_ID}",
        MODEL_PATH, quiet=False
    )

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model dimuat: input shape = {model.input_shape}")

# ================= EKSTRAKSI EMBEDDING =================
# ⚠️ PENTING: fungsi ini harus menghasilkan vektor 1024-dim
# yang SAMA PERSIS dengan cara kamu membuat X_embeddings.npy saat training
# Ganti isi fungsi ini sesuai preprocessing training kamu

def extract_embedding(audio_path: str) -> np.ndarray:
    """
    Ekstraksi embedding 1024-dim dari file audio.
    Sesuaikan dengan preprocessing saat membuat X_embeddings.npy
    """
    # Load audio — sama dengan saat training
    y, sr = librosa.load(audio_path, sr=16000, duration=10)

    # Contoh: MFCC 40 koef, flatten jadi 1024-dim
    # Ganti bagian ini kalau training pakai cara lain (YAMNet, VGGish, dll)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feat = mfcc.flatten()

    # Pad atau crop ke 1024 — sama dengan saat training
    if len(feat) < 1024:
        feat = np.pad(feat, (0, 1024 - len(feat)))
    else:
        feat = feat[:1024]

    return feat.reshape(1, -1).astype(np.float32)

# ================= ENDPOINT =================
@app.get("/")
def root():
    return {
        "status": "TBC API aktif",
        "model_input_shape": str(model.input_shape),
        "threshold": THRESHOLD
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Simpan file audio sementara di /tmp (wajib di HF Spaces)
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir="/tmp"
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Ekstraksi embedding → prediksi
        embedding = extract_embedding(tmp_path)

        # Validasi dimensi sebelum prediksi
        expected_dim = model.input_shape[1]  # 1024
        if embedding.shape[1] != expected_dim:
            return {
                "error": f"Dimensi embedding tidak cocok: "
                         f"dapat {embedding.shape[1]}, butuh {expected_dim}"
            }

        prob  = float(model.predict(embedding, verbose=0)[0][0])
        label = "TBC" if prob > THRESHOLD else "Sehat"
        conf  = prob if label == "TBC" else (1 - prob)

        return {
            "label": label,
            "confidence": round(conf * 100, 2),
            "probability_tbc": round(prob, 4),
            "threshold_used": THRESHOLD,
            "keterangan": (
                "Terindikasi TBC. Segera periksakan ke dokter."
                if label == "TBC"
                else "Tidak terindikasi TBC. Tetap jaga kesehatan."
            )
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        os.unlink(tmp_path)
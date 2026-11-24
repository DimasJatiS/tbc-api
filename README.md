---
title: TBC Audio Classifier API
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# ⚠️ Archived: use the Hugging Face Space instead

This repository is no longer maintained. The TBXpert app calls the API in the Hugging Face Space:

**https://huggingface.co/spaces/DImmmaas/TBC_AI**

Why it was retired:

- `main.py` computes placeholder MFCC features, while the model in this repo was trained on YAMNet embeddings. Its predictions are therefore not meaningful.
- `GDRIVE_ID` is never defined, so a fresh deploy fails at startup when the model file is missing.
- Keeping two copies of the serving code is how the train/serve mismatch happened. The maintained version lives in one place only.

The maintained Space uses HeAR embeddings and a participant-level evaluation. See its README for the model card.

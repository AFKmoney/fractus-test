# Fractus Space — Docker image for HuggingFace Spaces.
FROM python:3.10-slim

# System deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Space code (app.py + static/).
COPY app.py .
COPY static/ ./static/

# Fractus source + checkpoint are cloned/downloaded at runtime via env vars.
# Default paths assume the Space's /data mount for the KB.
ENV FRACTUS_PATH=/app/fractus_src \
    FRACTUS_CHECKPOINT=/data/fractus_latest.pt \
    FRACTUS_KB_PATH=/data/fractus_kb.pkl \
    PORT=7860

# Download Fractus source on container start (clone from GitHub).
RUN echo '#!/bin/sh\n\
if [ ! -d "$FRACTUS_PATH" ]; then\n\
  echo "[Fractus] Cloning source from GitHub..."\n\
  git clone --depth 1 https://github.com/AFKmoney/fractus.git "$FRACTUS_PATH"\n\
fi\n\
if [ ! -f "$FRACTUS_CHECKPOINT" ] && [ -n "$HF_TOKEN" ]; then\n\
  echo "[Fractus] Downloading checkpoint from HF..."\n\
  python3 -c "from huggingface_hub import hf_hub_download; import shutil; p = hf_hub_download(\"thefinalboss/Fractus\", \"checkpoints/fractus_1b_latest.pt\", repo_type=\"model\", token=\"$HF_TOKEN\"); shutil.copy(p, \"$FRACTUS_CHECKPOINT\")" || echo "[Fractus] Checkpoint download failed — using random weights"\n\
fi\n\
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}\n\
' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]

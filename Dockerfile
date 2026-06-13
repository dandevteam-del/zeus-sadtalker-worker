# Zeus SadTalker worker — RunPod serverless GPU image (built by RunPod from this repo).
# SadTalker (OpenTalker) animates a STILL portrait + audio into a talking head with
# real mouth motion + subtle head movement — the right tool for an AI-photo avatar
# (LatentSync is for re-syncing existing video). Code + deps only in the image; all
# model weights + caches live on the attached NETWORK VOLUME (/runpod-volume).
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SADTALKER_DIR=/app/SadTalker \
    HOME=/runpod-volume \
    HF_HOME=/runpod-volume/hf \
    TORCH_HOME=/runpod-volume/torch \
    TMPDIR=/runpod-volume/tmp

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-dev build-essential \
        git ffmpeg libgl1 libglib2.0-0 ca-certificates wget && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
# torch 2.1.0 cu121 runs on Ada (sm_89) + Ampere (sm_86) — our pinned GPU pool.
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --extra-index-url https://download.pytorch.org/whl/cu121

RUN git clone --depth 1 https://github.com/OpenTalker/SadTalker.git /app/SadTalker

# SadTalker's deps, pinned to a set that works with torch 2.1.
RUN pip install \
        numpy==1.23.5 face_alignment==1.3.5 imageio==2.19.3 imageio-ffmpeg==0.4.7 \
        librosa==0.10.1 numba resampy==0.4.2 pydub==0.25.1 scipy==1.10.1 \
        kornia==0.6.8 tqdm yacs==0.1.8 pyyaml joblib scikit-image==0.21.0 \
        basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8 av safetensors runpod requests

# basicsr (used by gfpgan) imports torchvision.transforms.functional_tensor, which
# was removed in torchvision>=0.17 — repoint it to the current module. Same for
# the well-known degradations.py import.
RUN python3 - <<'PY'
import os, basicsr, re
f = os.path.join(os.path.dirname(basicsr.__file__), "data", "degradations.py")
s = open(f).read().replace(
    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
    "from torchvision.transforms.functional import rgb_to_grayscale")
open(f, "w").write(s)
print("patched", f)
PY

ENV PYTHONPATH="/app/SadTalker:${PYTHONPATH}"
COPY handler.py /app/handler.py
CMD ["python3", "-u", "/app/handler.py"]

# Zeus SadTalker worker — RunPod serverless GPU image (built by RunPod from this repo).
# SadTalker (OpenTalker) animates a STILL portrait + audio into a talking head with
# real mouth motion + subtle head movement — the right tool for an AI-photo avatar
# (LatentSync is for re-syncing existing video). Code + deps only in the image; all
# model weights + caches live on the attached NETWORK VOLUME (/runpod-volume).
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# NOTE: do NOT point HOME/TMPDIR at /runpod-volume here — the network volume is
# only mounted at RUNTIME, not during the image build, so build-time apt/dpkg
# (ca-certificates postinst uses TMPDIR via mktemp) would fail. handler.py sets
# HOME/TMPDIR/HF_HOME onto the volume at runtime instead.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SADTALKER_DIR=/app/SadTalker

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

# SadTalker's deps. ORDER MATTERS: numpy first, then basicsr/gfpgan with
# --no-build-isolation (their setup.py imports torch+numpy; pip's default isolated
# build env can't see them → the classic "No module named 'torch'" build failure).
RUN pip install numpy==1.23.5
RUN pip install --no-build-isolation basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8
RUN pip install \
        face_alignment==1.3.5 imageio==2.19.3 imageio-ffmpeg==0.4.7 \
        librosa==0.10.1 numba==0.57.1 resampy==0.4.2 pydub==0.25.1 scipy==1.10.1 \
        kornia==0.6.8 tqdm yacs==0.1.8 pyyaml joblib scikit-image==0.19.3 \
        av safetensors runpod requests

# basicsr (used by gfpgan) imports torchvision.transforms.functional_tensor, which
# was removed in torchvision>=0.17 — repoint it to the current module. Single-line
# RUN (no heredoc) so it builds on any Docker builder, BuildKit or not.
RUN python3 -c "import os,basicsr; f=os.path.join(os.path.dirname(basicsr.__file__),'data','degradations.py'); s=open(f).read().replace('functional_tensor','functional'); open(f,'w').write(s); print('patched',f)"

# SadTalker's vendored code (e.g. src/face3d/util/my_awing_arch.py) still uses the
# np.float / np.int / np.bool aliases removed in NumPy>=1.24. Rewrite them to the
# builtins across the whole tree, word-boundary aware so np.float64/np.int32 survive.
COPY patch_np.py /app/patch_np.py
RUN python3 /app/patch_np.py /app/SadTalker

# Re-assert the numpy pin LAST so nothing in the batch above leaves a >=1.24 numpy
# (which would remove np.float and break 3DMM landmark extraction). Fail the build
# loudly if the alias is gone.
RUN pip install --force-reinstall --no-deps numpy==1.23.5 && \
    python3 -c "import numpy as np; print('numpy', np.__version__); assert hasattr(np,'float'), 'np.float missing — numpy got upgraded'"

ENV PYTHONPATH="/app/SadTalker:${PYTHONPATH}"
COPY handler.py /app/handler.py
CMD ["python3", "-u", "/app/handler.py"]

"""RunPod serverless handler — SadTalker talking-head from a STILL portrait.

Receives a source image + driven audio (base64), runs SadTalker, returns the
rendered talking-head mp4 (base64). Unlike LatentSync (which re-syncs an existing
video), SadTalker GENERATES mouth motion + head movement from one still image —
the correct engine for an AI-generated photo avatar.

Input  (event["input"]):
    image_b64        source portrait (png/jpg), base64        [required]
    audio_b64        voiceover (wav), base64                  [required]
    still            reduce head motion (presenter look), default True
    preprocess       crop|resize|full|extcrop|extfull, default "full"
    size             256 | 512, default 256 (GFPGAN enhances either way)
    enhancer         gfpgan | RestoreFormer | "" (none), default "gfpgan"
    expression_scale mouth/expression intensity, default 1.0
Output:
    video_b64        rendered talking-head mp4, base64
"""
import base64
import glob
import os
import subprocess
import time
import uuid

import runpod
import requests

SAD = os.environ.get("SADTALKER_DIR", "/app/SadTalker")
VOL = "/runpod-volume"
MODELS = f"{VOL}/sadtalker"                 # persistent weights on the volume
CKPT = f"{MODELS}/checkpoints"
GFP = f"{MODELS}/gfpgan/weights"
for d in (CKPT, GFP, f"{VOL}/tmp", f"{VOL}/hf", f"{VOL}/torch"):
    os.makedirs(d, exist_ok=True)
# Point caches/tmp at the volume at RUNTIME (the volume is mounted now). These are
# deliberately NOT Docker ENV — that would break the image build (volume absent).
os.environ.setdefault("HOME", VOL)
os.environ.setdefault("TMPDIR", f"{VOL}/tmp")
os.environ.setdefault("HF_HOME", f"{VOL}/hf")
os.environ.setdefault("TORCH_HOME", f"{VOL}/torch")

# SadTalker reads ./checkpoints and ./gfpgan/weights relative to its dir — point
# those at the volume so the (large) weights persist across cold starts.
def _link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            return
        import shutil
        shutil.rmtree(dst, ignore_errors=True) if os.path.isdir(dst) else os.remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)

_link(CKPT, f"{SAD}/checkpoints")
_link(f"{MODELS}/gfpgan", f"{SAD}/gfpgan")

REL = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
FX = "https://github.com/xinntao/facexlib/releases/download"
FILES = {
    f"{CKPT}/mapping_00109-model.pth.tar": f"{REL}/mapping_00109-model.pth.tar",
    f"{CKPT}/mapping_00229-model.pth.tar": f"{REL}/mapping_00229-model.pth.tar",
    f"{CKPT}/SadTalker_V0.0.2_256.safetensors": f"{REL}/SadTalker_V0.0.2_256.safetensors",
    f"{CKPT}/SadTalker_V0.0.2_512.safetensors": f"{REL}/SadTalker_V0.0.2_512.safetensors",
    f"{GFP}/alignment_WFLW_4HG.pth": f"{FX}/v0.1.0/alignment_WFLW_4HG.pth",
    f"{GFP}/detection_Resnet50_Final.pth": f"{FX}/v0.1.0/detection_Resnet50_Final.pth",
    f"{GFP}/parsing_parsenet.pth": f"{FX}/v0.2.2/parsing_parsenet.pth",
    f"{GFP}/GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
}
_models_ready = False


def _dl(dst, url):
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                tmp = dst + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                os.replace(tmp, dst)
            return
        except Exception as e:
            print(f"[dl] {os.path.basename(dst)} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def _ensure_models():
    global _models_ready
    if _models_ready:
        return
    for dst, url in FILES.items():
        if not os.path.exists(dst) or os.path.getsize(dst) < 10000:
            print(f"[models] fetching {os.path.basename(dst)} …", flush=True)
            _dl(dst, url)
    _models_ready = True
    print("[models] ready on volume", flush=True)


def handler(event):
    inp = event.get("input") or {}
    img_b64 = inp.get("image_b64")
    aud_b64 = inp.get("audio_b64")
    if not img_b64 or not aud_b64:
        return {"error": "image_b64 and audio_b64 are required"}

    try:
        _ensure_models()
    except Exception as e:
        return {"error": f"model download failed: {e}"}

    job = uuid.uuid4().hex[:8]
    work = f"{VOL}/tmp/{job}"
    os.makedirs(work, exist_ok=True)
    img_p = f"{work}/src.png"
    aud_p = f"{work}/audio.wav"
    out_d = f"{work}/out"
    with open(img_p, "wb") as f:
        f.write(base64.b64decode(img_b64))
    with open(aud_p, "wb") as f:
        f.write(base64.b64decode(aud_b64))

    size = str(int(inp.get("size", 256)))
    cmd = [
        "python3", "inference.py",
        "--source_image", img_p,
        "--driven_audio", aud_p,
        "--result_dir", out_d,
        "--checkpoint_dir", f"{SAD}/checkpoints",
        "--preprocess", str(inp.get("preprocess", "full")),
        "--size", size,
        "--expression_scale", str(inp.get("expression_scale", 1.0)),
    ]
    if inp.get("still", True):
        cmd.append("--still")
    enh = inp.get("enhancer", "gfpgan")
    if enh:
        cmd += ["--enhancer", enh]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=SAD, capture_output=True, text=True)
    tail = (proc.stdout or "")[-1500:] + "\n" + (proc.stderr or "")[-2500:]
    if proc.returncode != 0:
        return {"error": f"sadtalker failed (rc={proc.returncode})", "log": tail}

    mp4s = sorted(glob.glob(f"{out_d}/**/*.mp4", recursive=True), key=os.path.getmtime)
    if not mp4s:
        return {"error": "no mp4 produced", "log": tail}
    with open(mp4s[-1], "rb") as f:
        vid = base64.b64encode(f.read()).decode("ascii")
    return {"video_b64": vid, "seconds": round(time.time() - t0, 1), "size": size}


runpod.serverless.start({"handler": handler})

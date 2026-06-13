# zeus-sadtalker-worker

RunPod serverless worker: **SadTalker** talking-head from a **still portrait + audio**.

The correct engine for an AI-generated photo avatar — it *generates* mouth motion
and subtle head movement from one image, where LatentSync only re-syncs the mouth
on existing video (which smears on a static still).

**Input** `event.input`: `image_b64`, `audio_b64`, optional `still` (default true),
`preprocess` (`full`), `size` (256), `enhancer` (`gfpgan`), `expression_scale`.
**Output**: `video_b64` (mp4).

Weights download once to the attached **network volume** (`/runpod-volume/sadtalker`).
Deploy: connect this repo as a RunPod serverless endpoint, attach a network
volume (US-IL-1), GPU pool Ada/Ampere (cu121). Client: `modules/sadtalker_runpod.py`.

License: SadTalker is Apache-2.0; model weights for research/commercial per OpenTalker.

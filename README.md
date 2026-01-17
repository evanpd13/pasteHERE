# Zero-1-to-3 Novel View Synthesis

Generate ±30° novel views from a single image using Stable Zero123. This project provides a local Gradio UI to
preprocess a subject (background removal + centering) and synthesize new views with a GIF preview.

## Quick Start

```bash
./run.sh
```

The UI launches at http://127.0.0.1:7860 by default.

## CLI Options

```bash
python app.py --model-id kxic/stable-zero123 --device cuda --image-size 256
```

Key flags:

- `--model-id` to swap in a different Zero123-compatible model.
- `--device` to pin inference to `cuda`, `mps`, or `cpu`.
- `--host`/`--port` to change the server binding.
- `--share` to enable Gradio sharing.
- `--skip-preload` to launch the UI before the model loads.
- `--image-size` to choose 256 or 512 resolution (when supported by the model).

## Usage

1. Upload an image in the UI.
2. Tune rotation angle, steps, guidance, seed, and background removal settings.
3. Click **Generate Views** to synthesize the 3 frames.
4. Optionally export the frames + GIF to a directory of your choice.

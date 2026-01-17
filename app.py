#!/usr/bin/env python3
"""
Zero-1-to-3 Novel View Synthesis App
Generates ±30° rotated views from a single image using Stable Zero123.
"""

import argparse
import contextlib
import io
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageOps
from rembg import remove
import gradio as gr

# Add current directory to path for pipeline import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("zero123")

DEFAULT_MODEL_ID = "kxic/stable-zero123"
DEFAULT_IMAGE_SIZE = 256


def resolve_device(preferred: str | None = None) -> tuple[str, torch.dtype]:
    """Resolve device and dtype from preference and hardware availability."""
    if preferred:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda", torch.float16
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps", torch.float32
        if preferred == "cpu":
            return "cpu", torch.float32
        LOGGER.warning("Requested device '%s' unavailable; falling back to auto.", preferred)

    if torch.backends.mps.is_available():
        return "mps", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


DEVICE, DTYPE = resolve_device(os.getenv("ZERO123_DEVICE"))

LOGGER.info("Using device: %s", DEVICE)

@dataclass(frozen=True)
class AppConfig:
    model_id: str = DEFAULT_MODEL_ID
    image_size: int = DEFAULT_IMAGE_SIZE


# Global pipeline (loaded once)
PIPELINE = None
PIPELINE_CONFIG = AppConfig()


def maybe_enable_xformers(pipe) -> None:
    """Enable xFormers if available."""
    if DEVICE != "cuda":
        return
    try:
        pipe.enable_xformers_memory_efficient_attention()
        LOGGER.info("Enabled xFormers memory efficient attention.")
    except Exception as exc:
        LOGGER.warning("xFormers not available: %s", exc)


def load_pipeline():
    """Load the Zero123 pipeline by manually assembling components."""
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    LOGGER.info("Loading Stable Zero123 pipeline...")

    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline, CCProjection

    model_id = PIPELINE_CONFIG.model_id

    # Load each component separately
    LOGGER.info("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=DTYPE)

    LOGGER.info("  Loading image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=DTYPE)

    LOGGER.info("  Loading feature extractor...")
    feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")

    LOGGER.info("  Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=DTYPE)

    LOGGER.info("  Loading scheduler...")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    LOGGER.info("  Loading CC projection...")
    cc_projection = CCProjection.from_pretrained(model_id, subfolder="cc_projection")

    # Assemble the pipeline
    PIPELINE = Zero1to3StableDiffusionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=feature_extractor,
        cc_projection=cc_projection,
        requires_safety_checker=False,
    )

    PIPELINE = PIPELINE.to(DEVICE)
    PIPELINE.enable_attention_slicing()
    PIPELINE.enable_vae_slicing()
    maybe_enable_xformers(PIPELINE)

    LOGGER.info("Pipeline loaded successfully!")
    return PIPELINE


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from image using rembg."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    output = remove(image)
    return output


def preprocess_image(
    image: Image.Image,
    size: int = DEFAULT_IMAGE_SIZE,
    remove_bg: bool = True,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Preprocess image for Zero123: resize, center, add background."""
    image = ImageOps.exif_transpose(image)
    if remove_bg:
        image = remove_background(image)

    # Get the bounding box of non-transparent pixels
    if image.mode == "RGBA":
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)

    # Resize while maintaining aspect ratio
    w, h = image.size
    if w == 0 or h == 0:
        raise ValueError("Invalid image size after preprocessing.")
    scale = min(size / w, size / h) * 0.8  # 80% to leave margin
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create white background and paste centered
    result = Image.new("RGB", (size, size), background_color)
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2

    if image.mode == "RGBA":
        result.paste(image, (paste_x, paste_y), image)
    else:
        result.paste(image, (paste_x, paste_y))

    return result


def inference_autocast():
    """Return an autocast context manager when appropriate."""
    if DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=DTYPE)
    return contextlib.nullcontext()


def get_generator(seed: int | None):
    """Build a torch Generator for deterministic outputs."""
    if seed is None or seed < 0:
        return None
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    return generator


def generate_novel_view(
    processed_image: Image.Image,
    azimuth: float,
    polar: float = 0,
    num_steps: int = 75,
    guidance: float = 3.0,
    seed: int | None = None,
) -> Image.Image:
    """Generate a novel view at the specified angles."""
    pipe = load_pipeline()

    # Zero123 pose format: [polar_deg, azimuth_deg, distance]
    # polar: elevation angle (up/down)
    # azimuth: rotation around vertical axis (left/right)
    pose = [polar, azimuth, 0.0]

    generator = get_generator(seed)
    with torch.inference_mode(), inference_autocast():
        result = pipe(
            input_imgs=processed_image,
            prompt_imgs=processed_image,
            poses=[pose],
            height=PIPELINE_CONFIG.image_size,
            width=PIPELINE_CONFIG.image_size,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]

    return result


def generate_rotation_set(
    image: Image.Image,
    angle: float = 30,
    num_steps: int = 75,
    guidance: float = 3.0,
    seed: int | None = None,
    remove_bg: bool = True,
):
    """Generate 3 images: +angle, original (0), -angle rotations."""

    # Preprocess the image once (removes background)
    LOGGER.info("Preprocessing image...")
    processed = preprocess_image(image, size=PIPELINE_CONFIG.image_size, remove_bg=remove_bg)

    pipe = load_pipeline()

    results = []
    # Generate: +angle, 0 (original), -angle
    angles = [angle, 0, -angle]

    for i, az in enumerate(angles):
        if az == 0:
            # For center image, use the preprocessed version directly
            results.append(processed)
            LOGGER.info("Using original preprocessed image for azimuth=0°")
        else:
            LOGGER.info("Generating view at azimuth=%s°...", az)
            pose = [0, az, 0.0]  # [polar, azimuth, distance]
            generator = get_generator(seed + i if seed is not None and seed >= 0 else None)

            with torch.inference_mode(), inference_autocast():
                result = pipe(
                    input_imgs=processed,
                    prompt_imgs=processed,
                    poses=[pose],
                    height=PIPELINE_CONFIG.image_size,
                    width=PIPELINE_CONFIG.image_size,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    generator=generator,
                ).images[0]
            results.append(result)

    return results


def create_gif(images: list, duration: float = 0.5) -> bytes:
    """Create a GIF from a list of PIL images."""
    gif_buffer = io.BytesIO()
    frames = [np.array(img) for img in images]
    imageio.mimsave(gif_buffer, frames, format='GIF', duration=duration, loop=0)
    gif_buffer.seek(0)
    return gif_buffer.getvalue()


def export_images(images: list, output_dir: str, base_name: str = "view"):
    """Export images to the specified directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels = ["plus30", "original", "minus30"]
    saved_paths = []

    for img, label in zip(images, labels):
        filename = f"{base_name}_{label}.png"
        filepath = output_path / filename
        img.save(filepath)
        saved_paths.append(str(filepath))
        LOGGER.info("Saved: %s", filepath)

    # Also save the GIF
    gif_path = output_path / f"{base_name}_animation.gif"
    gif_data = create_gif(images)
    with open(gif_path, "wb") as f:
        f.write(gif_data)
    saved_paths.append(str(gif_path))
    LOGGER.info("Saved: %s", gif_path)

    return saved_paths


# Store generated images globally for export
CURRENT_IMAGES = []


def process_image(image, angle, num_steps, guidance, seed, remove_bg):
    """Main processing function for Gradio."""
    global CURRENT_IMAGES

    if image is None:
        return None, None, None, None, "Please upload an image first."

    try:
        # Generate the 3 rotated views
        images = generate_rotation_set(
            Image.fromarray(image),
            angle=angle,
            num_steps=int(num_steps),
            guidance=guidance,
            seed=int(seed) if seed is not None else None,
            remove_bg=remove_bg,
        )

        CURRENT_IMAGES = images

        # Create GIF preview in system temp directory
        gif_data = create_gif(images, duration=0.5)
        gif_path = os.path.join(tempfile.gettempdir(), "preview.gif")
        with open(gif_path, "wb") as f:
            f.write(gif_data)

        return images[0], images[1], images[2], gif_path, "Generation complete!"

    except Exception as e:
        LOGGER.exception("Generation failed.")
        return None, None, None, None, f"Error: {str(e)}"


def do_export(output_dir):
    """Export current images to directory."""
    global CURRENT_IMAGES

    if not CURRENT_IMAGES:
        return "No images to export. Generate images first."

    if not output_dir:
        return "Please specify an output directory."

    try:
        saved = export_images(CURRENT_IMAGES, output_dir)
        return f"Exported {len(saved)} files to {output_dir}"
    except Exception as e:
        return f"Export error: {str(e)}"


def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(title="Zero-1-to-3 Novel View Synthesis") as app:
        gr.Markdown("# Zero-1-to-3 Novel View Synthesis")
        gr.Markdown("Upload an image to generate ±30° rotated views using Stable Zero123.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Image", type="numpy")

                with gr.Group():
                    gr.Markdown("### Settings")
                    angle_slider = gr.Slider(
                        minimum=10, maximum=60, value=30, step=5,
                        label="Rotation Angle (±degrees)"
                    )
                    steps_slider = gr.Slider(
                        minimum=25, maximum=100, value=75, step=5,
                        label="Inference Steps (more = better quality)"
                    )
                    guidance_slider = gr.Slider(
                        minimum=1.0, maximum=10.0, value=3.0, step=0.5,
                        label="Guidance Scale"
                    )
                    seed_slider = gr.Slider(
                        minimum=-1, maximum=2147483647, value=-1, step=1,
                        label="Seed (-1 = random)"
                    )
                    remove_bg_checkbox = gr.Checkbox(
                        value=True,
                        label="Remove Background"
                    )

                generate_btn = gr.Button("Generate Views", variant="primary")

                with gr.Group():
                    gr.Markdown("### Export")
                    output_dir = gr.Textbox(
                        label="Export Directory",
                        placeholder="/path/to/output/folder"
                    )
                    export_btn = gr.Button("Export Images")

                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                with gr.Row():
                    img_plus = gr.Image(label="+30° View", type="pil")
                    img_center = gr.Image(label="Original (BG Removed)", type="pil")
                    img_minus = gr.Image(label="-30° View", type="pil")

                gif_preview = gr.Image(label="GIF Preview", type="filepath")

        # Connect events
        generate_btn.click(
            fn=process_image,
            inputs=[input_image, angle_slider, steps_slider, guidance_slider, seed_slider, remove_bg_checkbox],
            outputs=[img_plus, img_center, img_minus, gif_preview, status]
        )

        export_btn.click(
            fn=do_export,
            inputs=[output_dir],
            outputs=[status]
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-1-to-3 Novel View Synthesis")
    parser.add_argument("--model-id", default=os.getenv("ZERO123_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--device", default=os.getenv("ZERO123_DEVICE"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--skip-preload", action="store_true", help="Start UI without preloading the model.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, choices=[256, 512])
    return parser.parse_args()


def main():
    """Main entry point."""
    global DEVICE, DTYPE, PIPELINE_CONFIG
    args = parse_args()

    PIPELINE_CONFIG = AppConfig(model_id=args.model_id, image_size=args.image_size)
    DEVICE, DTYPE = resolve_device(args.device)

    LOGGER.info("=" * 50)
    LOGGER.info("Zero-1-to-3 Novel View Synthesis")
    LOGGER.info("=" * 50)
    LOGGER.info("Model: %s", PIPELINE_CONFIG.model_id)
    LOGGER.info("Device: %s", DEVICE)
    LOGGER.info("Image size: %s", PIPELINE_CONFIG.image_size)
    LOGGER.info("=" * 50)

    if not args.skip_preload:
        LOGGER.info("Loading model (this may take a moment)...")
        load_pipeline()

    LOGGER.info("Starting web UI at http://%s:%s", args.host, args.port)

    # Launch Gradio
    app = create_ui()
    app.launch(share=args.share, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()

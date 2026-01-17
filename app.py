#!/usr/bin/env python3
"""
Zero-1-to-3 Novel View Synthesis App
Generates ±30° rotated views from a single image using Stable Zero123.
"""

import os
import sys
import tempfile

# Add current directory to path for pipeline import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from rembg import remove
import gradio as gr
import imageio

# Determine device
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32  # MPS works better with float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"Using device: {DEVICE}")

# Global pipeline (loaded once)
PIPELINE = None


def load_pipeline():
    """Load the Zero123 pipeline by manually assembling components."""
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    print("Loading Stable Zero123 pipeline...")

    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline, CCProjection

    model_id = "kxic/stable-zero123"

    # Load each component separately
    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=DTYPE)

    print("  Loading image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=DTYPE)

    print("  Loading feature extractor...")
    feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")

    print("  Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=DTYPE)

    print("  Loading scheduler...")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    print("  Loading CC projection...")
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

    print("Pipeline loaded successfully!")
    return PIPELINE


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from image using rembg."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    output = remove(image)
    return output


def preprocess_image(image: Image.Image, size: int = 256) -> Image.Image:
    """Preprocess image for Zero123: resize, center, add white background."""
    # Remove background first
    image = remove_background(image)

    # Get the bounding box of non-transparent pixels
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # Resize while maintaining aspect ratio
    w, h = image.size
    scale = min(size / w, size / h) * 0.8  # 80% to leave margin
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create white background and paste centered
    result = Image.new("RGB", (size, size), (255, 255, 255))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2

    if image.mode == "RGBA":
        result.paste(image, (paste_x, paste_y), image)
    else:
        result.paste(image, (paste_x, paste_y))

    return result


def generate_novel_view(processed_image: Image.Image, azimuth: float, polar: float = 0,
                        num_steps: int = 75, guidance: float = 3.0) -> Image.Image:
    """Generate a novel view at the specified angles."""
    pipe = load_pipeline()

    # Zero123 pose format: [polar_deg, azimuth_deg, distance]
    # polar: elevation angle (up/down)
    # azimuth: rotation around vertical axis (left/right)
    pose = [polar, azimuth, 0.0]

    with torch.no_grad():
        result = pipe(
            input_imgs=processed_image,
            prompt_imgs=processed_image,
            poses=[pose],
            height=256,
            width=256,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
        ).images[0]

    return result


def generate_rotation_set(image: Image.Image, angle: float = 30,
                          num_steps: int = 75, guidance: float = 3.0):
    """Generate 3 images: +angle, original (0), -angle rotations."""

    # Preprocess the image once (removes background)
    print("Preprocessing image (removing background)...")
    processed = preprocess_image(image)

    pipe = load_pipeline()

    results = []
    # Generate: +angle, 0 (original), -angle
    angles = [angle, 0, -angle]

    for i, az in enumerate(angles):
        if az == 0:
            # For center image, use the preprocessed version directly
            results.append(processed)
            print(f"Using original preprocessed image for azimuth=0°")
        else:
            print(f"Generating view at azimuth={az}°...")
            pose = [0, az, 0.0]  # [polar, azimuth, distance]

            with torch.no_grad():
                result = pipe(
                    input_imgs=processed,
                    prompt_imgs=processed,
                    poses=[pose],
                    height=256,
                    width=256,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
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
        print(f"Saved: {filepath}")

    # Also save the GIF
    gif_path = output_path / f"{base_name}_animation.gif"
    gif_data = create_gif(images)
    with open(gif_path, "wb") as f:
        f.write(gif_data)
    saved_paths.append(str(gif_path))
    print(f"Saved: {gif_path}")

    return saved_paths


# Store generated images globally for export
CURRENT_IMAGES = []


def process_image(image, angle, num_steps, guidance):
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
            guidance=guidance
        )

        CURRENT_IMAGES = images

        # Create GIF preview in system temp directory
        gif_data = create_gif(images, duration=0.5)
        gif_path = os.path.join(tempfile.gettempdir(), "preview.gif")
        with open(gif_path, "wb") as f:
            f.write(gif_data)

        return images[0], images[1], images[2], gif_path, "Generation complete!"

    except Exception as e:
        import traceback
        traceback.print_exc()
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
            inputs=[input_image, angle_slider, steps_slider, guidance_slider],
            outputs=[img_plus, img_center, img_minus, gif_preview, status]
        )

        export_btn.click(
            fn=do_export,
            inputs=[output_dir],
            outputs=[status]
        )

    return app


def main():
    """Main entry point."""
    print("=" * 50)
    print("Zero-1-to-3 Novel View Synthesis")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print("Loading model (this may take a moment)...")
    print("=" * 50)

    # Pre-load the pipeline
    load_pipeline()

    print("Starting web UI at http://127.0.0.1:7860")

    # Launch Gradio
    app = create_ui()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()

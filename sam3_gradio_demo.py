#!/usr/bin/env python3
"""
SAM3 Interactive Vision Studio
English-only interactive image segmentation demo for SAM3.
"""

import io
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
import importlib.util

COLOR_PALETTE = [
    (244, 67, 54),
    (33, 150, 243),
    (76, 175, 80),
    (255, 193, 7),
    (156, 39, 176),
    (0, 188, 212),
    (255, 87, 34),
    (63, 81, 181),
]

# Add current directory to Python path so sam3 modules can be imported
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

sam3_available = (
    importlib.util.find_spec("sam3.model_builder") is not None
    and importlib.util.find_spec("sam3.model.sam3_image_processor") is not None
    and importlib.util.find_spec("sam3.visualization_utils") is not None
)

if sam3_available:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
else:
    print("SAM3 dependencies are missing; running in docs-only mode.")
    Sam3Processor = None  # type: ignore
    build_sam3_image_model = None  # type: ignore

# Global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def initialize_models():
    """Initialize the SAM3 image predictor if assets are available."""
    if Sam3Processor is None or build_sam3_image_model is None:
        print("SAM3 dependencies are missing; running in docs-only mode.")
        return None

    try:
        model_dir = current_dir / "models"
        checkpoint_path = model_dir / "sam3.pt"
        bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        missing_files = []
        if not checkpoint_path.exists():
            missing_files.append(str(checkpoint_path))
        if not bpe_path.exists():
            missing_files.append(str(bpe_path))

        if missing_files:
            print("Model assets are missing. The app will run with documentation-only features.")
            for file in missing_files:
                print(f" - Missing: {file}")
            return None

        image_model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path),
            device=DEVICE,
        )
        image_predictor = Sam3Processor(image_model, device=DEVICE)

        print("Image model initialized successfully.")
        return image_predictor

    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return None


# Global predictor instance
image_predictor = initialize_models()


def handle_image_click(
    img,
    original_img,
    detections_state,
    evt: gr.SelectData,
    mode,
    current_points,
    current_boxes,
    click_state,
):
    """Handle click events on the image and provide real-time visual feedback."""
    if img is None:
        return img, current_points, current_boxes, click_state, "Please upload an image first."

    base = render_overlays(original_img or img, detections_state or [])
    vis_img = np.array(base) if base is not None else (original_img or img).copy()

    x, y = evt.index
    x, y = int(x), int(y)

    info_msg = ""

    if mode == "📍 Point Prompt":
        new_point = f"{x},{y}"
        if current_points:
            current_points += f";{new_point}"
        else:
            current_points = new_point

        # Draw a red solid circle with white outline
        cv2.circle(vis_img, (x, y), 6, (255, 0, 0), -1)
        cv2.circle(vis_img, (x, y), 6, (255, 255, 255), 1)

        info_msg = f"✅ Added point: {new_point}"
        return vis_img, current_points, current_boxes, None, info_msg

    elif mode == "🔲 Box Prompt":
        if click_state is None:
            click_state = [x, y]
            cv2.circle(vis_img, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(vis_img, (x, y), 6, (255, 255, 255), 1)
            info_msg = f"🔵 Start recorded: {x},{y}. Click the opposite corner to finish."
            return vis_img, current_points, current_boxes, click_state, info_msg
        else:
            x1, y1 = click_state
            x2, y2 = x, y

            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1, x2)
            ymax = max(y1, y2)

            if xmin == xmax:
                xmax += 1
            if ymin == ymax:
                ymax += 1

            new_box = f"{xmin},{ymin},{xmax},{ymax}"
            if current_boxes:
                current_boxes += f";{new_box}"
            else:
                current_boxes = new_box

            cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

            info_msg = f"✅ Added box: {new_box}"
            return vis_img, current_points, current_boxes, None, info_msg

    return vis_img, current_points, current_boxes, click_state, info_msg


def segment_image(
    input_image,
    text_prompt,
    confidence_threshold,
    point_prompt,
    box_prompt,
    original_image=None,
    progress=gr.Progress(),
):
    """Perform image segmentation and return preview plus mask package."""
    image_to_process = original_image if original_image is not None else input_image

    empty_response = (None, "Please upload an image.", None, None, None)

    if image_to_process is None:
        return empty_response

    if not text_prompt and not point_prompt and not box_prompt:
        return (
            None,
            "Provide at least one prompt (text, point, or box).",
            None,
            None,
            None,
        )

    try:
        if image_predictor is None:
            return (
                None,
                "Model is not available. Upload assets to enable segmentation.",
                None,
                None,
                None,
            )

        start_time = time.time()
        progress(0.1, desc="Loading image...")

        if isinstance(image_to_process, np.ndarray):
            image = Image.fromarray(image_to_process)
        else:
            image = image_to_process

        state = image_predictor.set_image(image)
        progress(0.3, desc="Parsing prompts...")

        if text_prompt:
            state = image_predictor.set_text_prompt(text_prompt, state)

        if point_prompt:
            points = []
            for point_str in point_prompt.split(";"):
                if point_str:
                    try:
                        x, y = map(float, point_str.split(","))
                        points.append([x, y])
                    except ValueError:
                        continue

            if points:
                width, height = image.size
                normalized_points = []
                for x, y in points:
                    normalized_points.append([x / width, y / height])

                for point in normalized_points:
                    box_size = min(width, height) * 0.05
                    box_width = box_size / width
                    box_height = box_size / height
                    box = [point[0], point[1], box_width, box_height]
                    state = image_predictor.add_geometric_prompt(box, True, state)

        if box_prompt:
            boxes = []
            for box_str in box_prompt.split(";"):
                if box_str:
                    try:
                        x1, y1, x2, y2 = map(float, box_str.split(","))
                        boxes.append([x1, y1, x2, y2])
                    except ValueError:
                        continue

            if boxes:
                width, height = image.size
                normalized_boxes = []
                for x1, y1, x2, y2 in boxes:
                    center_x = (x1 + x2) / 2 / width
                    center_y = (y1 + y2) / 2 / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    normalized_boxes.append([center_x, center_y, box_width, box_height])

                for box in normalized_boxes:
                    state = image_predictor.add_geometric_prompt(box, True, state)

        state = image_predictor.set_confidence_threshold(confidence_threshold, state)

        progress(0.7, desc="Running inference...")

        if "boxes" in state and len(state["boxes"]) > 0:
            scores = state["scores"].detach().cpu().numpy()
            boxes_np = state["boxes"].detach().cpu().numpy()
            masks = state.get("masks")
            masks_np = masks.detach().cpu().numpy() if masks is not None else None

            return (
                image,
                boxes_np,
                masks_np,
                scores,
                time.time() - start_time,
            )

            fd, output_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            result_image.save(output_path)

            detection_rows = []
            if "scores" in state and "boxes" in state:
                scores = state["scores"].detach().cpu().numpy()
                boxes_np = state["boxes"].detach().cpu().numpy()
                for idx, (score, box) in enumerate(zip(scores, boxes_np)):
                    x1, y1, x2, y2 = box
                    label = text_prompt or f"Object {idx + 1}"
                    detection_rows.append(
                        [idx + 1, label, round(float(score), 3), f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}"]
                    )

            segmentation_zip = None
            if "masks" in state:
                masks = state["masks"].detach().cpu().numpy()
                boxes_np = state["boxes"].detach().cpu().numpy()
                scores = state["scores"].detach().cpu().numpy()
                temp_dir = Path(tempfile.mkdtemp(prefix="sam3_masks_"))
                manifest = []
                for idx, (mask, box, score) in enumerate(zip(masks, boxes_np, scores)):
                    mask_array = (mask.squeeze() * 255).astype(np.uint8)
                    mask_image = Image.fromarray(mask_array)
                    mask_path = temp_dir / f"mask_{idx + 1}.png"
                    mask_image.save(mask_path)
                    manifest.append(
                        {
                            "id": idx + 1,
                            "label": text_prompt or f"Object {idx + 1}",
                            "score": float(score),
                            "box": [float(x) for x in box.tolist()],
                            "mask": mask_path.name,
                        }
                    )

                manifest_path = temp_dir / "manifest.json"
                with open(manifest_path, "w", encoding="utf-8") as mf:
                    import json

                    json.dump({"instances": manifest}, mf, indent=2)

                zip_fd, zip_path = tempfile.mkstemp(suffix=".zip")
                os.close(zip_fd)
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for file_path in temp_dir.iterdir():
                        zf.write(file_path, arcname=file_path.name)
                segmentation_zip = zip_path

            return result_image, info, output_path, detection_rows, segmentation_zip
        else:
            return (
                image,
                None,
                [],
                None,
            )

    except Exception as e:
        return None, f"❌ Processing failed: {str(e)}", None, [], None


def create_demo():
    """Create the Gradio demo interface."""

    custom_css = """
    .container { max-width: 1800px; width: 95%; margin: auto; padding-top: 20px; }
    .gradio-container { max-width: unset; }
    h1 { text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2d3748; margin-bottom: 10px; }
    .description { text-align: center; font-size: 1.1em; color: #4a5568; margin-bottom: 30px; }
    .gr-button-primary { background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); border: none; }
    .gr-box { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    #interaction-info { font-weight: bold; color: #2b6cb0; text-align: center; background-color: #ebf8ff; padding: 10px; border-radius: 5px; border: 1px solid #bee3f8; }
    .main-row { gap: 16px; align-items: stretch; }
    .control-card { background: #0f172a0d; border-radius: 12px; padding: 16px; border: 1px solid #e2e8f0; }
    .seg-card { background: white; border-radius: 12px; padding: 12px; border: 1px solid #e2e8f0; }
    .accordion-compact .gr-panel { padding: 8px 12px; }
    .mode-radio .wrap { display: flex; width: 100%; gap: 10px; }
    .mode-radio .wrap label { flex: 1; justify-content: center; text-align: center; }
    """

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

    with gr.Blocks(theme=theme, css=custom_css, title="SAM3 Interactive Vision Studio") as demo:
        with gr.Column(elem_classes="container"):
            gr.Markdown("# 👁️ SAM3 Interactive Vision Studio")
            gr.Markdown(
                "Next-gen SAM3 image segmentation playground. Upload model files to enable inference; "
                "otherwise, explore the instructions below.",
                elem_classes="description",
            )

            with gr.Tabs():
                with gr.TabItem("🖼️ Image Segmentation", id="tab_image"):
                    with gr.Row(elem_classes="main-row"):
                        with gr.Column(scale=0.95, elem_classes="control-card"):
                            gr.Markdown("### Upload & prompts")
                            image_canvas = gr.Image(
                                type="numpy",
                                label="Upload, click to prompt, and view results",
                                elem_id="input_image",
                                interactive=True,
                                height=650,
                            )
                            interaction_info = gr.Markdown(
                                "👆 點擊圖片加入 point 或 box，再按下 segment 以覆蓋顯示。",
                                elem_id="interaction-info",
                            )
                            image_info = gr.Textbox(
                                label="📊 Result summary", interactive=False, lines=2
                            )

                            original_image_state = gr.State(None)
                            click_state = gr.State(None)

                            with gr.Accordion(
                                "🎛️ Prompt & threshold settings", open=False, elem_classes="accordion-compact"
                            ):
                                interaction_mode = gr.Radio(
                                    choices=["📍 Point Prompt", "🔲 Box Prompt"],
                                    value="📍 Point Prompt",
                                    label="Interaction mode",
                                    elem_classes="mode-radio",
                                )
                                confidence_threshold = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.4,
                                    step=0.05,
                                    label="🎯 Confidence threshold",
                                )

                            with gr.Accordion("📝 Advanced prompt options", open=False):
                                text_prompt = gr.Textbox(
                                    label="Text prompt",
                                    placeholder="Describe what to segment, e.g., 'a red car' or 'a cat'",
                                    lines=1,
                                )

                                with gr.Row():
                                    example_text_btn = gr.Button("🐱 Cat", size="sm")
                                    example_point_btn = gr.Button("📍 Sample point", size="sm")

                                with gr.Row(visible=False):
                                    point_prompt = gr.Textbox(label="Point coordinates")
                                    box_prompt = gr.Textbox(label="Box coordinates")

                            with gr.Row():
                                segment_button = gr.Button(
                                    "🚀 Run segmentation", variant="primary", size="lg"
                                )
                                clear_prompts_btn = gr.Button(
                                    "🗑️ Reset", size="sm", variant="secondary"
                                )

                            interaction_info = gr.Markdown(
                                "👆 Click the image to add points or box corners.",
                                elem_id="interaction-info",
                            )

                            with gr.Accordion("📦 Detected objects", open=True):
                                detected_objects = gr.Dataframe(
                                    headers=["ID", "Name", "Confidence", "Box (x1,y1,x2,y2)"],
                                    value=[],
                                )

                            with gr.Accordion("📥 Downloads", open=True):
                                download_output = gr.File(label="Segmentation preview (PNG)")
                                segmentation_package = gr.File(
                                    label="Segmentation package (ZIP of masks)",
                                    file_types=[".zip"],
                                )

                        with gr.Column(scale=1.55, elem_classes="seg-card"):
                            image_output = gr.Image(
                                type="numpy", label="✨ Segmentation preview", height=650
                            )
                            image_info = gr.Textbox(
                                label="📊 Result summary", interactive=False, lines=2
                            )

                    def store_original_image(img):
                        if img is None:
                            return (
                                None,
                                None,
                                [],
                                1,
                                "",
                                "",
                                None,
                                "Upload an image to begin.",
                                [],
                                gr.Dropdown.update(choices=[], value=None),
                                None,
                                None,
                                "",
                            )

                        return (
                            img,
                            img,
                            [],
                            1,
                            "",
                            "",
                            None,
                            "Image loaded. Add prompts, then press segment to render.",
                            [],
                            gr.Dropdown.update(choices=[], value=None),
                            None,
                            None,
                            "",
                        )

                    image_canvas.upload(
                        fn=store_original_image,
                        inputs=[image_canvas],
                        outputs=[
                            image_canvas,
                            original_image_state,
                            detections_state,
                            next_id_state,
                            point_prompt,
                            box_prompt,
                            click_state,
                            interaction_info,
                            detected_objects,
                            detection_selector,
                            segmentation_package,
                            download_output,
                            image_info,
                        ],
                    )

                    image_canvas.select(
                        fn=handle_image_click,
                        inputs=[
                            image_canvas,
                            original_image_state,
                            detections_state,
                            interaction_mode,
                            point_prompt,
                            box_prompt,
                            click_state,
                        ],
                        outputs=[image_canvas, point_prompt, box_prompt, click_state, interaction_info],
                    )

                    def reset_all(orig_img):
                        if orig_img is None:
                            return (
                                None,
                                orig_img,
                                [],
                                1,
                                "",
                                "",
                                None,
                                "Upload an image to begin.",
                                [],
                                gr.Dropdown.update(choices=[], value=None),
                                None,
                                None,
                                "",
                            )

                        return (
                            orig_img,
                            orig_img,
                            [],
                            1,
                            "",
                            "",
                            None,
                            "♻️ Reset complete. Prompts and detections cleared.",
                            [],
                            gr.Dropdown.update(choices=[], value=None),
                            None,
                            None,
                            "",
                        )

                    clear_prompts_btn.click(
                        fn=reset_all,
                        inputs=[original_image_state],
                        outputs=[
                            image_canvas,
                            original_image_state,
                            detections_state,
                            next_id_state,
                            point_prompt,
                            box_prompt,
                            click_state,
                            interaction_info,
                            detected_objects,
                            detection_selector,
                            segmentation_package,
                            download_output,
                            image_info,
                        ],
                    )

                    def run_segmentation(
                        img,
                        text_prompt,
                        confidence_threshold,
                        point_prompt,
                        box_prompt,
                        original_img,
                        detections,
                        next_id,
                    ):
                        base_image = original_img if original_img is not None else img
                        if base_image is None:
                            return (
                                img,
                                original_img,
                                detections,
                                next_id,
                                point_prompt,
                                box_prompt,
                                None,
                                "Please upload an image first.",
                                build_detection_rows(detections),
                                gr.Dropdown.update(choices=[d.get("id") for d in detections], value=None),
                                build_segmentation_package(detections),
                                None,
                                "",
                            )

                        result_image, boxes_np, masks_np, scores, elapsed = segment_image(
                            base_image,
                            text_prompt,
                            confidence_threshold,
                            point_prompt,
                            box_prompt,
                            original_img,
                        )

                        if boxes_np is None or scores is None:
                            overlay = render_overlays(base_image, detections)
                            return (
                                overlay,
                                original_img,
                                detections,
                                next_id,
                                "",
                                "",
                                None,
                                "⚠️ No objects detected. Adjust prompts or threshold.",
                                build_detection_rows(detections),
                                gr.Dropdown.update(choices=[d.get("id") for d in detections], value=None),
                                build_segmentation_package(detections),
                                None,
                                "",
                            )

                        new_detections = list(detections)
                        for idx, (box, score) in enumerate(zip(boxes_np, scores)):
                            mask = None
                            if masks_np is not None and len(masks_np) > idx:
                                mask = masks_np[idx].squeeze() > 0.5

                            label = text_prompt or f"Object {next_id}"
                            color = COLOR_PALETTE[(next_id - 1) % len(COLOR_PALETTE)]
                            new_detections.append(
                                {
                                    "id": next_id,
                                    "label": label,
                                    "score": float(score),
                                    "box": [float(x) for x in box.tolist()],
                                    "mask": mask,
                                    "color": color,
                                }
                            )
                            next_id += 1

                        overlay = render_overlays(base_image, new_detections)

                        fd, output_path = tempfile.mkstemp(suffix=".png")
                        os.close(fd)
                        if overlay is not None:
                            Image.fromarray(np.array(overlay)).save(output_path)
                        else:
                            output_path = None

                        info_msg = (
                            f"✨ Segmentation complete. Added {len(new_detections) - len(detections)} "
                            f"object(s). Total: {len(new_detections)} | Time: {elapsed:.2f}s"
                        )

                        selector_update = gr.Dropdown.update(
                            choices=[det.get("id") for det in new_detections], value=None
                        )

                        return (
                            overlay,
                            original_img,
                            new_detections,
                            next_id,
                            "",
                            "",
                            None,
                            info_msg,
                            build_detection_rows(new_detections),
                            selector_update,
                            build_segmentation_package(new_detections),
                            output_path,
                            info_msg,
                        )

                    segment_button.click(
                        fn=run_segmentation,
                        inputs=[
                            image_canvas,
                            text_prompt,
                            confidence_threshold,
                            point_prompt,
                            box_prompt,
                            original_image_state,
                            detections_state,
                            next_id_state,
                        ],
                        outputs=[
                            image_canvas,
                            original_image_state,
                            detections_state,
                            next_id_state,
                            point_prompt,
                            box_prompt,
                            click_state,
                            interaction_info,
                            detected_objects,
                            detection_selector,
                            segmentation_package,
                            download_output,
                            image_info,
                        ],
                    )

                    def delete_detection(det_id, original_img, detections):
                        if not det_id:
                            return (
                                render_overlays(original_img, detections),
                                original_img,
                                detections,
                                gr.Dropdown.update(choices=[d.get("id") for d in detections], value=None),
                                build_detection_rows(detections),
                                build_segmentation_package(detections),
                                None,
                                "Select an ID to delete.",
                            )

                        remaining = [d for d in detections if str(d.get("id")) != str(det_id)]
                        overlay = render_overlays(original_img, remaining)

                        fd, output_path = tempfile.mkstemp(suffix=".png")
                        os.close(fd)
                        if overlay is not None:
                            Image.fromarray(np.array(overlay)).save(output_path)
                        else:
                            output_path = None

                        selector_update = gr.Dropdown.update(
                            choices=[d.get("id") for d in remaining], value=None
                        )

                        return (
                            overlay,
                            original_img,
                            remaining,
                            selector_update,
                            build_detection_rows(remaining),
                            build_segmentation_package(remaining),
                            output_path,
                            f"Detection {det_id} removed.",
                        )

                    delete_detection_btn.click(
                        fn=delete_detection,
                        inputs=[detection_selector, original_image_state, detections_state],
                        outputs=[
                            image_canvas,
                            original_image_state,
                            detections_state,
                            detection_selector,
                            detected_objects,
                            segmentation_package,
                            download_output,
                            interaction_info,
                        ]
                    )

                    example_text_btn.click(fn=lambda: "a cat", outputs=[text_prompt])
                    example_point_btn.click(fn=lambda: "100,100", outputs=[point_prompt])

        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #718096; font-size: 0.9em;">
                Powered by <strong>SAM3</strong> | 2025 SAM3 Interactive Studio
            </div>
            """
        )

    return demo


def main():
    """Main entrypoint."""
    model_dir = current_dir / "models"
    model_dir.mkdir(exist_ok=True)

    checkpoint_path = model_dir / "sam3.pt"
    bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    if not checkpoint_path.exists() or not bpe_path.exists():
        print("⚠️ Model assets are missing. Segmentation will be disabled until files are added.")
        print(f"Expected files:\n1. {checkpoint_path}\n2. {bpe_path}")

    print("🚀 Launching SAM3 Interactive Vision Studio...")
    demo = create_demo()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False,
        debug=True,
        allowed_paths=[str(current_dir)],
    )


if __name__ == "__main__":
    main()

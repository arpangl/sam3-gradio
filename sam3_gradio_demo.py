#!/usr/bin/env python3
"""
SAM3 Interactive Vision Studio
Interactive image segmentation and video tracking system powered by SAM3
"""

import os
import sys
import time
import io
import numpy as np
import torch
import gradio as gr
from PIL import Image
import cv2
from pathlib import Path
import tempfile
import json

# Add the current directory to the Python path so we can import sam3 modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import SAM3 modules
try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    from sam3.model.data_misc import FindStage
    from sam3.visualization_utils import plot_results, visualize_formatted_frame_output, render_masklet_frame
    from sam3.model import box_ops
except ImportError as e:
    print(f"Failed to import SAM3 modules: {e}")
    print("Please ensure SAM3 dependencies are installed correctly")
    sys.exit(1)

# Global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Initialize models
def initialize_models():
    """Initialize SAM3 image and video predictors"""
    try:
        # Check that required model files exist
        model_dir = current_dir / "models"
        checkpoint_path = model_dir / "sam3.pt"
        bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        if not checkpoint_path.exists():
            print(f"Model checkpoint not found: {checkpoint_path}")
            print("Please download the SAM3 model file to the models directory")
            return None, None

        if not bpe_path.exists():
            print(f"BPE vocabulary file not found: {bpe_path}")
            return None, None

        # Initialize image model
        image_model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path),
            device=DEVICE
        )

        # Create image processor
        image_predictor = Sam3Processor(image_model, device=DEVICE)

        # Initialize video predictor
        video_predictor = Sam3VideoPredictor(
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path)
        )

        print("Models initialized successfully")
        return image_predictor, video_predictor

    except Exception as e:
        print(f"Model initialization failed: {e}")
        return None, None

# Global predictor instances
image_predictor, video_predictor = initialize_models()

def handle_image_click(img, original_img, evt: gr.SelectData, mode, current_points, current_boxes, click_state):
    """Handle image clicks and provide real-time visual feedback"""
    if img is None:
        return img, current_points, current_boxes, click_state, "Please upload an image first"

    # If no original image is stored, use the current image
    if original_img is None:
        original_img = img.copy()

    # Draw on the currently displayed image
    vis_img = img.copy()
    
    x, y = evt.index
    x, y = int(x), int(y)
    
    info_msg = ""
    
    if mode == "📍 Point Prompt":
        new_point = f"{x},{y}"
        if current_points:
            current_points += f";{new_point}"
        else:
            current_points = new_point

        # Draw a red dot on the image
        cv2.circle(vis_img, (x, y), 6, (255, 0, 0), -1) # Red filled circle
        cv2.circle(vis_img, (x, y), 6, (255, 255, 255), 1) # White outline

        info_msg = f"✅ Added point: {new_point}"
        return vis_img, current_points, current_boxes, None, info_msg

    elif mode == "🔲 Box Prompt":
        if click_state is None:
            # First click - draw the start point (blue)
            click_state = [x, y]
            cv2.circle(vis_img, (x, y), 6, (0, 0, 255), -1) # Blue filled circle
            cv2.circle(vis_img, (x, y), 6, (255, 255, 255), 1) # White outline
            info_msg = f"🔵 Start point recorded: {x},{y}. Click the opposite corner to finish the box."
            return vis_img, current_points, current_boxes, click_state, info_msg
        else:
            # Second click - draw the box (green)
            x1, y1 = click_state
            x2, y2 = x, y
            
            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1, x2)
            ymax = max(y1, y2)
            
            # Ensure the box has a size
            if xmin == xmax: xmax += 1
            if ymin == ymax: ymax += 1
            
            new_box = f"{xmin},{ymin},{xmax},{ymax}"
            if current_boxes:
                current_boxes += f";{new_box}"
            else:
                current_boxes = new_box
            
            # Draw a green rectangle
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
    progress=gr.Progress()
):
    """Image segmentation pipeline"""
    # Use the original image if available; otherwise fall back to the displayed image
    image_to_process = original_image if original_image is not None else input_image

    if image_to_process is None:
        return None, "Please upload an image"

    if not text_prompt and not point_prompt and not box_prompt:
        return None, "Please provide at least one prompt (text, point, or box)"

    try:
        if image_predictor is None:
            return None, "Model is not initialized. Please check the model files."

        start_time = time.time()
        progress(0.1, desc="Loading image...")

        # Convert image format
        if isinstance(image_to_process, np.ndarray):
            image = Image.fromarray(image_to_process)
        else:
            image = image_to_process

        # Set image
        state = image_predictor.set_image(image)
        progress(0.3, desc="Parsing prompts...")

        # Handle text prompt
        if text_prompt:
            state = image_predictor.set_text_prompt(text_prompt, state)

        # Handle point prompts
        if point_prompt:
            points = []
            for point_str in point_prompt.split(';'):
                if point_str:
                    try:
                        x, y = map(float, point_str.split(','))
                        points.append([x, y])
                    except ValueError:
                        continue
                        
            if points:
                width, height = image.size
                normalized_points = []
                for x, y in points:
                    normalized_points.append([x/width, y/height])
                    
                for point in normalized_points:
                    box_size = min(width, height) * 0.05
                    box_width = box_size / width
                    box_height = box_size / height
                    box = [point[0], point[1], box_width, box_height]
                    state = image_predictor.add_geometric_prompt(box, True, state)
        
        # Handle box prompts
        if box_prompt:
            boxes = []
            for box_str in box_prompt.split(';'):
                if box_str:
                    try:
                        x1, y1, x2, y2 = map(float, box_str.split(','))
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
        
        # Set the confidence threshold
        state = image_predictor.set_confidence_threshold(confidence_threshold, state)

        progress(0.7, desc="Running model inference...")
        
        # Retrieve results
        if "boxes" in state and len(state["boxes"]) > 0:
            # Visualize results
            import matplotlib.pyplot as plt
            
            # Use the official plot_results helper to draw masks, boxes, and scores
            plot_results(image, state)

            # Convert the created figure to a PIL image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            result_image = Image.open(buf)
            plt.close() # Close figure to release memory
            
            processing_time = time.time() - start_time
            info = f"✨ Processing complete | Time: {processing_time:.2f}s | Detected {len(state['boxes'])} objects"

            return result_image, info
        else:
            return image, "⚠️ No objects detected. Try adjusting prompts or lowering the confidence threshold."

    except Exception as e:
        return None, f"❌ Processing failed: {str(e)}"
def convert_output_format(outputs):
    """Convert model outputs to the visualization-friendly format"""
    if not outputs: return {}
    
    # Simplified conversion logic, reusing the core idea
    if "out_binary_masks" in outputs:
        formatted_outputs = {
            "out_boxes_xywh": [], "out_probs": [], "out_obj_ids": [], "out_binary_masks": []
        }
        
        masks = outputs["out_binary_masks"]
        if not isinstance(masks, (list, np.ndarray)): masks = [masks]
        formatted_outputs["out_binary_masks"] = list(masks)
        
        if "out_obj_ids" in outputs: formatted_outputs["out_obj_ids"] = list(outputs["out_obj_ids"])
        else: formatted_outputs["out_obj_ids"] = list(range(len(masks)))
            
        if "out_probs" in outputs: formatted_outputs["out_probs"] = list(outputs["out_probs"])
        else: formatted_outputs["out_probs"] = [1.0] * len(masks)

        if "out_boxes_xywh" in outputs:
            formatted_outputs["out_boxes_xywh"] = list(outputs["out_boxes_xywh"])
        else:
            # Compute bounding boxes
            for mask in formatted_outputs["out_binary_masks"]:
                if isinstance(mask, np.ndarray) and mask.any():
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if rows.any() and cols.any():
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        h, w = mask.shape
                        formatted_outputs["out_boxes_xywh"].append([x_min/w, y_min/h, (x_max-x_min)/w, (y_max-y_min)/h])
                    else: formatted_outputs["out_boxes_xywh"].append([0,0,0,0])
                else: formatted_outputs["out_boxes_xywh"].append([0,0,0,0])
        return formatted_outputs
        
    # Fallback logic omitted for brevity as it mirrors the previous implementation
    # The essential mask handling is kept for robustness
    elif "masks" in outputs:
         formatted_outputs = {"out_boxes_xywh": [], "out_probs": [], "out_obj_ids": [], "out_binary_masks": []}
         masks = outputs["masks"]
         # Handle list or tensor
         if not isinstance(masks, list) and hasattr(masks, 'shape'):
             if len(masks.shape) == 4: masks = [m[0] for m in masks.cpu().numpy()]
             elif len(masks.shape) == 3: masks = [m for m in masks.cpu().numpy()]
         
         for i, mask in enumerate(masks):
             if hasattr(mask, 'shape') and len(mask.shape) > 2: mask = mask.squeeze()
             formatted_outputs["out_binary_masks"].append(mask)
             formatted_outputs["out_obj_ids"].append(i)
             formatted_outputs["out_probs"].append(1.0)
             # Simple box calculation
             if isinstance(mask, np.ndarray) and mask.any():
                 h, w = mask.shape
                 y, x = np.where(mask)
                 formatted_outputs["out_boxes_xywh"].append([x.min()/w, y.min()/h, (x.max()-x.min())/w, (y.max()-y.min())/h])
             else: formatted_outputs["out_boxes_xywh"].append([0,0,0,0])
         return formatted_outputs
         
    return {}

def process_video(
    input_video,
    text_prompt,
    confidence_threshold,
    progress=gr.Progress()
):
    """Video processing pipeline"""
    if input_video is None:
        return None, "Please upload a video"

    if not text_prompt:
        return None, "Please provide a text prompt"

    try:
        if video_predictor is None:
            return None, "Model is not initialized. Please check the model files."

        start_time = time.time()
        progress(0.1, desc="Reading video...")
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fd, output_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened(): return None, "Unable to open the video file"
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            progress(0.2, desc="Starting tracking session...")
            session_response = video_predictor.start_session(resource_path=input_video)
            session_id = session_response["session_id"]

            progress(0.3, desc="Applying prompt...")
            video_predictor.add_prompt(session_id=session_id, frame_idx=0, text=text_prompt)

            progress(0.4, desc="Tracking targets...")
            outputs_per_frame = {}
            for response in video_predictor.handle_stream_request(
                request={"type": "propagate_in_video", "session_id": session_id}
            ):
                outputs_per_frame[response["frame_index"]] = response["outputs"]
            
            for frame_idx in range(frame_count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if frame_idx in outputs_per_frame:
                    formatted_outputs = convert_output_format(outputs_per_frame[frame_idx])
                    if formatted_outputs.get("out_binary_masks"):
                        vis_frame = render_masklet_frame(
                            img=frame_rgb,
                            outputs=formatted_outputs,
                            frame_idx=frame_idx,
                            alpha=0.5
                        )
                    else: vis_frame = frame_rgb
                else: vis_frame = frame_rgb
                
                vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                out.write(vis_frame_bgr)
                
                progress_value = 0.4 + 0.5 * (frame_idx / frame_count)
                progress(progress_value, desc=f"Rendering frame {frame_idx+1}/{frame_count}")
            
            cap.release()
            out.release()
            video_predictor.close_session(session_id)
            
            processing_time = time.time() - start_time
            info = f"✨ Processing complete | Time: {processing_time:.2f}s | Total frames: {frame_count}"

            return str(output_path), info

    except Exception as e:
        return None, f"❌ Processing failed: {str(e)}"

def create_demo():
    """Create a polished Gradio demo UI"""

    # Custom CSS
    custom_css = """
    .container { max-width: 1200px; margin: auto; padding-top: 20px; }
    h1 { text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2d3748; margin-bottom: 10px; }
    .description { text-align: center; font-size: 1.1em; color: #4a5568; margin-bottom: 30px; }
    .gr-button-primary { background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); border: none; }
    .gr-box { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    #interaction-info { font-weight: bold; color: #2b6cb0; text-align: center; background-color: #ebf8ff; padding: 10px; border-radius: 5px; border: 1px solid #bee3f8; }
    
    /* Make the radio button group stretch horizontally */
    .mode-radio .wrap { display: flex; width: 100%; gap: 10px; }
    .mode-radio .wrap label { flex: 1; justify-content: center; text-align: center; }
    """

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif']
    )

    with gr.Blocks(theme=theme, css=custom_css, title="SAM3 Interactive Vision Studio") as demo:
        
        with gr.Column(elem_classes="container"):
            gr.Markdown("# 👁️ SAM3 Interactive Vision Studio")
            gr.Markdown("Next-generation image segmentation and video tracking with SAM3", elem_classes="description")
            
            with gr.Tabs():
                # ================= Image segmentation tab =================
                with gr.TabItem("🖼️ Smart Image Segmentation", id="tab_image"):
                    with gr.Row():
                        # Left control column
                        with gr.Column(scale=1):
                            image_input = gr.Image(type="numpy", label="Input Image (click to interact)", elem_id="input_image")

                            # Store the original image state
                            original_image_state = gr.State(None)
                            click_state = gr.State(None)
                            
                            with gr.Group():
                                gr.Markdown("### 🎮 Interaction Mode")
                                # Mode selection
                                interaction_mode = gr.Radio(
                                    choices=["📍 Point Prompt", "🔲 Box Prompt"],
                                    value="📍 Point Prompt",
                                    label="Choose a mode",
                                    show_label=False,
                                    elem_classes="mode-radio"
                                )
                                # Clear prompts button
                                with gr.Row():
                                    clear_prompts_btn = gr.Button("🗑️ Clear Prompts", size="sm", variant="secondary")

                                interaction_info = gr.Markdown("👆 Click the image to start adding prompts...", elem_id="interaction-info")
                            
                            with gr.Accordion("📝 Advanced Prompt Options", open=True):
                                text_prompt = gr.Textbox(
                                    label="Text Prompt",
                                    placeholder="Describe the object, e.g., 'a red car' or 'a cat'",
                                    lines=1
                                )

                                with gr.Row():
                                    gr.Markdown("Quick examples:")
                                    example_text_btn = gr.Button("🐱 Cat", size="sm")
                                    example_point_btn = gr.Button("📍 Sample Point", size="sm")

                                with gr.Row(visible=False): # Hide raw coordinate inputs while keeping backend logic intact
                                    point_prompt = gr.Textbox(label="Point coordinates")
                                    box_prompt = gr.Textbox(label="Box coordinates")
                            
                            confidence_threshold = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.4, step=0.05,
                                label="🎯 Confidence Threshold"
                            )

                            segment_button = gr.Button("🚀 Run Segmentation", variant="primary", size="lg")

                        # Right results column
                        with gr.Column(scale=1):
                            image_output = gr.Image(type="numpy", label="✨ Segmentation Result")
                            image_info = gr.Textbox(label="📊 Analysis Summary", interactive=False, lines=2)
                    
                    # Event bindings

                    # 1. Save the original image on upload
                    def store_original_image(img): return img, None # Reset click state
                    image_input.upload(fn=store_original_image, inputs=[image_input], outputs=[original_image_state, click_state])

                    # 2. Handle image clicks
                    image_input.select(
                        fn=handle_image_click,
                        inputs=[image_input, original_image_state, interaction_mode, point_prompt, box_prompt, click_state],
                        outputs=[image_input, point_prompt, box_prompt, click_state, interaction_info]
                    )

                    # 3. Clear prompts
                    def clear_prompts(orig_img):
                        if orig_img is None: return None, "", "", None, "Please upload an image first"
                        return orig_img, "", "", None, "♻️ Prompts cleared and image reset"
                    clear_prompts_btn.click(
                        fn=clear_prompts,
                        inputs=[original_image_state],
                        outputs=[image_input, point_prompt, box_prompt, click_state, interaction_info]
                    )

                    # 4. Segmentation button
                    segment_button.click(
                        fn=segment_image,
                        inputs=[image_input, text_prompt, confidence_threshold, point_prompt, box_prompt, original_image_state],
                        outputs=[image_output, image_info]
                    )

                    # 5. Example buttons
                    example_text_btn.click(fn=lambda: "a cat", outputs=[text_prompt])
                    example_point_btn.click(fn=lambda: "100,100", outputs=[point_prompt])

                # ================= Video tracking tab =================
                with gr.TabItem("🎬 Video Object Tracking", id="tab_video"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            video_input = gr.Video(label="📂 Upload Video File")
                            
                            with gr.Group():
                                video_text_prompt = gr.Textbox(
                                    label="📝 Target description",
                                    placeholder="e.g., 'a person running' (currently only the first-frame text prompt is supported)",
                                    lines=2
                                )
                                video_confidence_threshold = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                                    label="🎯 Tracking confidence"
                                )
                            
                            process_button = gr.Button("▶️ Start Tracking", variant="primary", size="lg")

                        with gr.Column(scale=1):
                            video_output = gr.Video(label="✨ Tracking Result")
                            video_info = gr.Textbox(label="📊 Processing Summary", interactive=False)
                    
                    process_button.click(
                        fn=process_video,
                        inputs=[video_input, video_text_prompt, video_confidence_threshold],
                        outputs=[video_output, video_info]
                    )
        
        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #718096; font-size: 0.9em;">
            Powered by <strong>SAM3</strong> | 2025 SAM3 Interactive Studio
        </div>
        """)
    
    return demo

def main():
    """Main entry point"""
    # Check that model files exist
    model_dir = current_dir / "models"
    if not model_dir.exists():
        print(f"Creating model directory: {model_dir}")
        model_dir.mkdir(exist_ok=True)
        
    checkpoint_path = model_dir / "sam3.pt"
    bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    
    if not checkpoint_path.exists() or not bpe_path.exists():
        print("⚠️ Model files are missing")
        print(f"Please ensure the following files exist:\n1. {checkpoint_path}\n2. {bpe_path}")

        response = input("Would you like to download the model files automatically? (y/n): ").lower().strip()
        if response == 'y':
            try:
                import download_models
                download_models.main()
            except Exception as e:
                print(f"Automatic download failed: {e}")
                return
        else:
            return

    print("🚀 Launching the SAM3 Interactive Vision Studio...")
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False,
        debug=True,
        allowed_paths=[str(current_dir)]
    )

if __name__ == "__main__":
    main()

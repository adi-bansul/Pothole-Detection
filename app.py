# --- START OF FILE app.py ---

import streamlit as st
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import time
from PIL import Image
import pandas as pd
import os
import tempfile

st.set_page_config(layout="wide")

# --- Configuration ---
MODEL_PATH = "best.pt"
RESULTS_CSV_PATH = "runs/results.csv"
DEFAULT_VIDEO_PATH = "sample/vid.mp4"
DEFAULT_IMAGE_PATH = "sample/img.jpg"
WEBCAM_DEFAULT_WIDTH = 640 # Default size for webcam frames
WEBCAM_DEFAULT_HEIGHT = 480
# --- End Configuration ---

@st.cache_resource # Cache the model loading
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure 'best.pt' is in the correct location.")
        st.stop()
    try:
        model = YOLO(model_path)
        _ = model.model.names # Access names to ensure they're loaded if needed later
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()

@st.cache_resource # Cache the annotator initialization
def get_annotators():
    tracker = sv.ByteTrack()
    mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, opacity=0.4)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.WHITE,
        text_scale=0.5,
        text_thickness=1,
        text_padding=2,
        text_position=sv.Position.TOP_CENTER
    )
    return tracker, mask_annotator, label_annotator

@st.cache_data # Cache the results loading
def load_validation_results(csv_path):
    if not os.path.exists(csv_path):
        st.warning(f"Training results file not found at {csv_path}. Cannot display validation metrics.")
        return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        if df.empty:
            st.warning(f"Results file {csv_path} is empty.")
            return None
        latest_results = df.iloc[-1]
        metrics = {
            "mAP50-95(B)": latest_results.get("metrics/mAP50-95(B)", "N/A"),
            "mAP50(B)": latest_results.get("metrics/mAP50(B)", "N/A"),
            "Precision(B)": latest_results.get("metrics/precision(B)", "N/A"),
            "Recall(B)": latest_results.get("metrics/recall(B)", "N/A"),
            "mAP50-95(M)": latest_results.get("metrics/mAP50-95(M)", "N/A"),
            "mAP50(M)": latest_results.get("metrics/mAP50(M)", "N/A"),
        }
        for key, value in metrics.items():
            if pd.notna(value) and isinstance(value, (int, float)):
                metrics[key] = f"{value:.3f}"
            elif pd.isna(value):
                 metrics[key] = "N/A"
        return metrics
    except KeyError as e:
        st.error(f"Missing expected column in {csv_path}: {e}. Check the CSV header.")
        return None
    except Exception as e:
        st.error(f"Error reading results file {csv_path}: {e}")
        return None

# Initialize model and annotators
model = load_model(MODEL_PATH)
tracker, mask_annotator, label_annotator = get_annotators()

# --- Processing & Display Functions ---

def process_frame(frame: np.ndarray, confidence: float) -> tuple[np.ndarray, sv.Detections]:
    results = model(frame, conf=confidence)[0]
    detections = sv.Detections.from_ultralytics(results)
    class_names = model.model.names
    labels = []
    if detections.confidence is not None and detections.class_id is not None:
        labels = [
            f"{class_names[class_id]}: {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]
    annotated_frame = mask_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )
    return annotated_frame, detections

def display_inference_metrics(detections: sv.Detections, col_metrics):
    num_detections = len(detections)
    avg_confidence = 0.0
    if num_detections > 0 and detections.confidence is not None:
        confidences = [c for c in detections.confidence if c is not None]
        if confidences:
             avg_confidence = np.mean(confidences)
    if len(col_metrics) >= 2:
        with col_metrics[0]:
            st.metric("Detected Potholes", num_detections)
        with col_metrics[1]:
            st.metric("Avg. Confidence", f"{avg_confidence:.2f}")
    else:
        st.metric("Detected Potholes", num_detections)
        st.metric("Avg. Confidence", f"{avg_confidence:.2f}")

# --- Input Source Specific Functions ---

def video_input(data_src, confidence, uploaded_file_obj):
    vid_file_path = None
    temp_file_to_delete = None

    if data_src == 'Sample data':
        if not os.path.exists(DEFAULT_VIDEO_PATH):
            st.error(f"Sample video not found at {DEFAULT_VIDEO_PATH}")
            st.info(f"Please make sure '{DEFAULT_VIDEO_PATH}' exists or select another source.")
            return
        vid_file_path = DEFAULT_VIDEO_PATH
        st.sidebar.write(f"Using sample: {os.path.basename(vid_file_path)}")
    elif data_src == 'Upload your own data':
        if uploaded_file_obj is None:
             st.info("☝️ Upload or drag & drop a video file using the sidebar.")
             return
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_obj.name)[1])
            tfile.write(uploaded_file_obj.read())
            vid_file_path = tfile.name
            temp_file_to_delete = vid_file_path
            st.sidebar.info(f"Processing uploaded: {uploaded_file_obj.name}")
            tfile.close()
        except Exception as e:
            st.error(f"Error handling uploaded video file: {e}")
            if temp_file_to_delete and os.path.exists(temp_file_to_delete):
                 try: os.remove(temp_file_to_delete)
                 except: pass
            return
    else:
        # This case should not happen with the current radio button setup
        st.error("Invalid data source selected for video.")
        return

    if vid_file_path:
        cap = None
        try:
            cap = cv2.VideoCapture(vid_file_path)
            if not cap.isOpened():
                st.error(f"Error opening video file: {vid_file_path}")
                return

            custom_size = st.sidebar.checkbox("Custom frame size", key="vid_custom_size")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            input_fps = cap.get(cv2.CAP_PROP_FPS)

            if custom_size:
                width = st.sidebar.number_input("Width", min_value=120, step=20, value=width, key="vid_width")
                height = st.sidebar.number_input("Height", min_value=120, step=20, value=height, key="vid_height")

            st.subheader("Video Detection")
            col_vis, col_data = st.columns([3, 1])

            with col_data:
                st.markdown("##### Frame Info")
                fps_metric = st.empty()
                size_metric_w = st.empty()
                size_metric_h = st.empty()
                st.markdown("---")
                st.markdown("##### Current Frame Detections")
                metric_placeholders = [st.empty(), st.empty()]

            with col_vis:
                output_image = st.empty()
                if st.button("Stop Video", key="stop_video"):
                    st.info("Stopping video processing.")
                    # This button click will cause Streamlit to rerun, breaking the loop naturally.
                    # No explicit break needed here if logic depends on rerun.

            prev_time = time.time()
            processing_fps = 0

            while cap.isOpened(): # Loop while capture is open
                ret, frame = cap.read()
                if not ret:
                    st.write("Video stream ended or read error.")
                    break

                frame_resized = cv2.resize(frame, (width, height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                processed_frame, detections = process_frame(frame_rgb, confidence)
                output_image.image(processed_frame, caption="Pothole Detection Result", use_column_width=True)

                curr_time = time.time()
                elapsed = curr_time - prev_time
                prev_time = curr_time
                processing_fps = 1 / elapsed if elapsed > 0 else 0

                display_inference_metrics(detections, metric_placeholders)
                fps_metric.metric("Processing FPS", f"{processing_fps:.1f} (Input: {input_fps:.1f})")
                size_metric_w.metric("Width", width)
                size_metric_h.metric("Height", height)
                # Re-run check: if button was clicked, loop will exit on next iteration due to rerun

        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")
            import traceback
            st.error(traceback.format_exc())
        finally:
            if cap is not None:
                cap.release()
            if temp_file_to_delete and os.path.exists(temp_file_to_delete):
                 try:
                     os.remove(temp_file_to_delete)
                 except Exception as e:
                     st.warning(f"Could not remove temporary file {temp_file_to_delete}: {e}")
            # Clear placeholders on exit or error
            if 'output_image' in locals() and hasattr(output_image, 'empty'): output_image.empty()
            if 'fps_metric' in locals() and hasattr(fps_metric, 'empty'): fps_metric.empty()
            if 'size_metric_w' in locals() and hasattr(size_metric_w, 'empty'): size_metric_w.empty()
            if 'size_metric_h' in locals() and hasattr(size_metric_h, 'empty'): size_metric_h.empty()
            if 'metric_placeholders' in locals():
                for p in metric_placeholders:
                    if hasattr(p, 'empty'): p.empty()


def image_input(data_src, confidence, uploaded_file_obj):
    img_array = None
    display_name = "Sample Image"

    if data_src == 'Sample data':
        if not os.path.exists(DEFAULT_IMAGE_PATH):
            st.error(f"Sample image not found at {DEFAULT_IMAGE_PATH}")
            st.info(f"Please make sure '{DEFAULT_IMAGE_PATH}' exists or select another source.")
            return
        img_array = cv2.imread(DEFAULT_IMAGE_PATH)
        if img_array is None:
             st.error(f"Could not read sample image file: {DEFAULT_IMAGE_PATH}")
             return
        display_name = os.path.basename(DEFAULT_IMAGE_PATH)
        st.sidebar.write(f"Using sample: {display_name}")
    elif data_src == 'Upload your own data':
        if uploaded_file_obj is None:
            st.info("☝️ Upload or drag & drop an image file using the sidebar.")
            return
        try:
            file_bytes = np.asarray(bytearray(uploaded_file_obj.read()), dtype=np.uint8)
            img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_array is None:
                st.error(f"Could not decode uploaded image: {uploaded_file_obj.name}.")
                return
            display_name = uploaded_file_obj.name
            st.sidebar.info(f"Processing uploaded: {display_name}")
        except Exception as e:
            st.error(f"Error reading uploaded image file: {e}")
            return
    else:
        # This case should not happen
        st.error("Invalid data source selected for image.")
        return

    if img_array is not None:
        try:
            image_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            h, w, _ = img_array.shape

            st.subheader(f"Image Detection Results: {display_name}")
            col_orig, col_proc, col_metrics = st.columns([2, 2, 1])

            with col_orig:
                st.image(image_rgb, caption="Original Image", use_column_width=True)

            start_time = time.time()
            processed_image, detections = process_frame(image_rgb, confidence)
            end_time = time.time()
            processing_time = end_time - start_time

            with col_proc:
                st.image(processed_image, caption="Pothole Detection Result (with Confidence)", use_column_width=True)

            with col_metrics:
                 st.markdown("##### Detection Info")
                 metric_placeholders = [st.empty(), st.empty()]
                 display_inference_metrics(detections, metric_placeholders)
                 st.metric("Processing Time (s)", f"{processing_time:.3f}")
                 st.metric("Image Size (HxW)", f"{h} x {w}")

        except Exception as e:
            st.error(f"An error occurred during image processing: {e}")
            import traceback
            st.error(traceback.format_exc())

# --- NEW Webcam Function ---
def webcam_input(confidence):
    st.subheader("Webcam Live Detection")
    st.sidebar.info("Attempting to access webcam...")

    cap = None
    try:
        # Try to open the default webcam (usually 0)
        cap = cv2.VideoCapture(0) # Use 0 for default webcam
        if not cap.isOpened():
            st.error("Cannot open webcam. Please ensure it's connected and not used by another application.")
            st.sidebar.error("Webcam access failed.")
            return

        st.sidebar.success("Webcam accessed successfully!")

        custom_size = st.sidebar.checkbox("Custom frame size", value=False, key="webcam_custom_size")
        width = WEBCAM_DEFAULT_WIDTH
        height = WEBCAM_DEFAULT_HEIGHT

        # Set camera resolution if possible (might not work on all cameras/OS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read actual width/height after attempting to set
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=actual_width, key="webcam_width")
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=actual_height, key="webcam_height")
        else:
             width = actual_width
             height = actual_height


        col_vis, col_data = st.columns([3, 1])

        with col_data:
            st.markdown("##### Frame Info")
            fps_metric = st.empty()
            size_metric_w = st.empty()
            size_metric_h = st.empty()
            st.markdown("---")
            st.markdown("##### Current Frame Detections")
            metric_placeholders = [st.empty(), st.empty()]

        with col_vis:
            # Use a button to stop the webcam feed
            stop_button_pressed = st.button("Stop Webcam", key="stop_webcam")
            output_image = st.empty()


        prev_time = time.time()
        processing_fps = 0

        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("Webcam frame read error or stream ended.")
                break

            # Flip the frame horizontally (mirror effect) - common for webcams
            frame = cv2.flip(frame, 1)

            # Resize frame if custom size is specified OR if it differs from target
            if (width != frame.shape[1] or height != frame.shape[0]):
                 frame_resized = cv2.resize(frame, (width, height))
            else:
                 frame_resized = frame

            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            processed_frame, detections = process_frame(frame_rgb, confidence)
            output_image.image(processed_frame, caption="Webcam Pothole Detection", use_column_width=True)

            curr_time = time.time()
            elapsed = curr_time - prev_time
            prev_time = curr_time
            processing_fps = 1 / elapsed if elapsed > 0 else 0

            display_inference_metrics(detections, metric_placeholders)
            fps_metric.metric("Processing FPS", f"{processing_fps:.1f}")
            size_metric_w.metric("Width", width)
            size_metric_h.metric("Height", height)

            # Check the button state again within the loop for faster stopping
            # Streamlit re-runs the script on button press, so this loop will naturally exit.
            # We mainly need the button variable to prevent the loop from starting again after release.

    except Exception as e:
        st.error(f"An error occurred during webcam processing: {e}")
        import traceback
        st.error(traceback.format_exc())
    finally:
        if cap is not None:
            cap.release()
            st.sidebar.info("Webcam released.")
        # Clear placeholders when stopped or on error
        if 'output_image' in locals() and hasattr(output_image, 'empty'): output_image.empty()
        if 'fps_metric' in locals() and hasattr(fps_metric, 'empty'): fps_metric.empty()
        if 'size_metric_w' in locals() and hasattr(size_metric_w, 'empty'): size_metric_w.empty()
        if 'size_metric_h' in locals() and hasattr(size_metric_h, 'empty'): size_metric_h.empty()
        if 'metric_placeholders' in locals():
            for p in metric_placeholders:
                if hasattr(p, 'empty'): p.empty()
        if 'stop_button_pressed' in locals() and stop_button_pressed:
             st.info("Webcam stopped.")


def main():
    st.title(" Pothole Detection System")
    st.sidebar.title("⚙️ Settings")

    # --- Input Selection ---
    input_type = st.sidebar.radio(
        "Select input type:",
        ['Image', 'Video', 'Webcam'], # Added Webcam type
        key='input_type_radio',
        # Automatically select Webcam if Webcam source is chosen
        index=2 if st.session_state.get('input_source_radio') == 'Webcam' else (1 if st.session_state.get('input_type_radio') == 'Video' else 0)
    )

    # --- Source Selection ---
    # Adjust source options based on type
    source_options = ['Sample data', 'Upload your own data']
    if input_type == 'Webcam':
        source_options = ['Webcam'] # Only allow Webcam source if type is Webcam
        st.session_state['input_source_radio'] = 'Webcam' # Force source to Webcam

    input_source = st.sidebar.radio(
        "Select input source:",
        options=source_options,
        key='input_source_radio'
     )

    # --- Confidence Slider ---
    confidence = st.sidebar.slider(
        'Confidence Threshold',
        min_value=0.1, max_value=1.0, value=0.3, step=0.05,
        key='confidence_slider'
    )

    st.sidebar.markdown("---")

    # --- File Uploader (Conditional) ---
    uploaded_file_obj = None
    if input_source == 'Upload your own data' and input_type != 'Webcam': # Only show if Upload is selected and type is not Webcam
        st.sidebar.markdown("### 📤 Upload File")
        if input_type == 'Image':
            uploaded_file_obj = st.sidebar.file_uploader(
                "Upload an image (or drag & drop)", type=['png', 'jpeg', 'jpg'], key='img_uploader_widget'
            )
        elif input_type == 'Video':
             uploaded_file_obj = st.sidebar.file_uploader(
                "Upload a video (or drag & drop)", type=['mp4', 'mov', 'avi', 'mkv', 'webm'], key='vid_uploader_widget'
            )
        st.sidebar.markdown("---") # Add separator after uploader if shown
    elif input_source == 'Webcam':
         st.sidebar.markdown("### 📷 Webcam Active")
         st.sidebar.write("_Settings below control the live feed._")
         st.sidebar.markdown("---")


    # --- Model Performance (Always shown) ---
    st.sidebar.markdown("### 📈 Model Performance")
    st.sidebar.markdown(
        "_Metrics from validation dataset (general quality). Not accuracy on current input._"
    )
    validation_metrics = load_validation_results(RESULTS_CSV_PATH)
    if validation_metrics:
        mcol1, mcol2 = st.sidebar.columns(2)
        with mcol1:
            st.metric("mAP50-95 (Mask)", validation_metrics.get('mAP50-95(M)', 'N/A'))
            st.metric("Precision (Box)", validation_metrics.get('Precision(B)', 'N/A'))
        with mcol2:
             st.metric("mAP50 (Mask)", validation_metrics.get('mAP50(M)', 'N/A'))
             st.metric("Recall (Box)", validation_metrics.get('Recall(B)', 'N/A'))
    else:
        st.sidebar.info("Validation results file not found or unreadable.")

    st.sidebar.markdown("---")


    # --- Main Area Execution Logic ---
    if input_type == 'Webcam' or input_source == 'Webcam': # Trigger webcam if either is selected
        webcam_input(confidence)
    elif input_type == 'Video':
        video_input(input_source, confidence, uploaded_file_obj)
    elif input_type == 'Image':
        image_input(input_source, confidence, uploaded_file_obj)
    else:
        st.info("Select an input type and source from the sidebar.")


if __name__ == "__main__":
    # Initialize session state if keys don't exist
    if 'input_type_radio' not in st.session_state:
        st.session_state['input_type_radio'] = 'Image' # Default
    if 'input_source_radio' not in st.session_state:
        st.session_state['input_source_radio'] = 'Sample data' # Default

    main()
# --- END OF FILE app.py ---
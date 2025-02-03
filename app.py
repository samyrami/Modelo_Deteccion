import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import torch
import logging
import sys
import psutil
import gc
import urllib.request


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

@st.cache_resource
def load_model():
    """Load YOLO model with memory management and error handling"""
    try:
        logger.info("Starting model load")
        
        # Verificar si el modelo existe, si no, descargarlo
        model_path = "yolov8n.pt"
        if not os.path.exists(model_path):
            logger.info("Downloading YOLOv8n model...")
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
            urllib.request.urlretrieve(url, model_path)
            logger.info("Model downloaded successfully")
        
        # Optimizar para inferencia
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_path)
        model.to(device)
        
        # Configurar para inferencia
        if device == "cuda":
            model.fuse()
        
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise
    
    
def process_image(image, model, confidence, show_boxes, show_labels):
    # Reducir aún más la resolución para mejorar FPS
    max_size = 416  # Reducido de 640 a 416 para mayor velocidad
    height, width = image.shape[:2]
    scale = min(max_size/width, max_size/height)
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Procesar imagen con half precision si hay GPU disponible
    with torch.no_grad():  # Desactivar gradientes para inferencia más rápida
        results = model(image, conf=confidence, verbose=False)
    
    # Dibujar resultados de manera más eficiente
    if show_boxes or show_labels:
        annotated_frame = results[0].plot(line_width=1)  # Líneas más delgadas para mejor rendimiento
    else:
        annotated_frame = image
    
    return annotated_frame, results[0]

def main():
    # Page configuration
    st.set_page_config(
        page_title="Object Detector",
        layout="wide"
    )
    
    try:
        logger.info("Starting application")
        
        st.title("Object Detector 🎥")
        
        # Load model with error handling
        try:
            with st.spinner("Loading model (this may take a minute)..."):
                model = load_model()
                logger.info("Model loaded successfully in main")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            st.error("Failed to initialize the model. Please try again later.")
            return
    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")

    # Mode selector
    mode = st.radio("Select mode:", ["Real-time Camera", "Upload file"], key="mode_selector")
    
    # Sidebar configurations
    st.sidebar.title("Settings")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="confidence_slider")
    
    # Checkboxes for labels and boxes
    show_labels = st.sidebar.checkbox("Show Labels", True, key="show_labels")
    show_boxes = st.sidebar.checkbox("Show Boxes", True, key="show_boxes")
    
    # Object counter
    object_counter = {}
    
    if mode == "Real-time Camera":
        # Video container
        video_placeholder = st.empty()
        
        # Camera state
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
            
        # Camera toggle button
        if st.button(
            "Start Camera" if not st.session_state.camera_running else "Stop Camera",
            key="camera_toggle"
        ):
            st.session_state.camera_running = not st.session_state.camera_running
            
        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0)
            # Configurar la cámara para mejor rendimiento
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)  # Establecer FPS objetivo
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir el buffer
            
            try:
                frame_skip = 0  # Contador para saltar frames
                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Error accessing camera")
                        break
                    
                    # Procesar solo cada 2 frames para mejorar la fluidez
                    frame_skip += 1
                    if frame_skip % 2 != 0:
                        continue
                    
                    # Usar la función de procesamiento optimizada
                    processed_frame, result = process_image(
                        frame, model, confidence, show_boxes, show_labels
                    )
                    
                    # Actualizar contadores de manera más eficiente
                    if frame_skip % 4 == 0:  # Actualizar contadores con menos frecuencia
                        object_counter.clear()
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            class_name = model.names[cls]
                            object_counter[class_name] = object_counter.get(class_name, 0) + 1
                    
                    # Mostrar frame de manera más eficiente
                    video_placeholder.image(
                        cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )
                    
                    # Update sidebar counter
                    st.sidebar.subheader("Detected Objects:")
                    for obj, count in object_counter.items():
                        st.sidebar.text(f"{obj}: {count}")
                    
            finally:
                cap.release()
    
    else:  # File upload mode
        uploaded_file = st.file_uploader(
            "Upload an image or video", 
            type=['jpg', 'jpeg', 'png', 'mp4'],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                if uploaded_file.type.startswith('image'):
                    image = cv2.imread(tmp_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    results = model.predict(image, conf=confidence)
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = model.names[cls]
                            
                            if class_name in object_counter:
                                object_counter[class_name] += 1
                            else:
                                object_counter[class_name] = 1
                            
                            if show_boxes:
                                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            if show_labels:
                                label = f"{class_name} {conf:.2f}"
                                cv2.putText(image, label, (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    st.image(image, caption='Result', use_container_width=True)
            
            finally:
                os.unlink(tmp_path)
            
            st.sidebar.subheader("Detected Objects:")
            for obj, count in object_counter.items():
                st.sidebar.text(f"{obj}: {count}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
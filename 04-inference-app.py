import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import torch


# Load the YOLO models

# detect_model_path = "best_weights/yolo11x-freeze-backbone.pt"
# segment_model_path = "best_weights/yolo11l-seg-freeze-backbone.pt"

detect_model_path = "best_weights/yolo11l-freeze-backbone.pt"
segment_model_path = "best_weights/yolo11l-seg-freeze-backbone.pt"


# Set-up CUDA device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# use a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"]="6"

# Use GPU for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the YOLO models using ultralytics
detect_model = YOLO(detect_model_path, task = 'detect')
segment_model = YOLO(segment_model_path, task = 'segment')

class_names = [
  'Bottled Soda',  
  'Cheese',        
  'Chocolate',             
  'Instant Coffee',
  'Condensed Milk',
  'Cooking Oil',
  'Corned Beef',
  'Garlic',
  'Instant Noodles',
  'Ketchup',
  'Lemon',
  'All Purpose Cream',
  'Mayonnaise',
  'Peanut Butter',
  'Pasta',
  'Canned Juice',
  'Crackers',
  'Canned Sardines',
  'Bottled Shampoo',
  'Soap',
  'Soy Sauce',
  'Toothpaste',
  'Canned Tuna',
  'Ethyl Alcohol'
]

last_frame_time = time.time()

# Define the function to process the webcam input for detection
def detection(image, conf_threshold=0.3):
    global last_frame_time

    if image is None or image.size == 0:
        return image, 0.0

    # start_time = time.time()
    
    # Convert the image to RGB format
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = detect_model(frame_rgb, conf=conf_threshold, imgsz=[1088,1920])
    
    # Make the image writable
    image = image.copy()
    
    # Postprocess the outputs
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            if conf > conf_threshold:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_names[cls]}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - last_frame_time)
    last_frame_time = end_time
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    return image

# Define the function to process the webcam input for segmentation
def segmentation(image, conf_threshold=0.3):
    global last_frame_time
    
    if image is None or image.size == 0:
        return image, 0.0

    # Convert the image to RGB format
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = segment_model(frame_rgb, conf=conf_threshold, imgsz=[1088,1920])
    
    # Make the image writable
    image = image.copy()
    
    # Postprocess the outputs
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            if conf > conf_threshold:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_names[cls]}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        masks = result.masks
        if masks is not None:
            for mask in masks.data:
                mask = mask.cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                # Define the mask color and transparency
                color = np.array([0, 0, 255], dtype=np.uint8)  # Blue color for the mask
                alpha = 0.5  # Set the transparency level (0.0 = fully transparent, 1.0 = fully opaque)
                
                # Create a colored mask only where the binary mask is active
                for c in range(3):  # Iterate over color channels
                    image[:, :, c] = np.where(mask == 1,
                                              (1 - alpha) * image[:, :, c] + alpha * color[c],
                                              image[:, :, c])

    end_time = time.time()
    fps = 1 / (end_time - last_frame_time)
    last_frame_time = end_time
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    return image

# Define the function to process the image based on the selected mode
def process_image(image, conf_threshold, mode):
    if mode == "Detection":
        return detection(image, conf_threshold)
    else:
        return segmentation(image, conf_threshold)

css = """.my-group {max-width: 800px !important; max-height: 800px !important;}
         .my-column {display: flex !important; justify-content: center !important; align-items: center !important;}"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Real-time Detection and Segmentation of Common Grocery Items
        </h1>
        """
    )
    with gr.Column():
        with gr.Row():
            conf_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.50,
            )
            mode = gr.Radio(
                choices=["Detection", "Segmentation"],
                value="Detection",
                label="Mode"
            )
        with gr.Row():
            input_img = gr.Image(sources=["webcam"], type="numpy", streaming=True)

    input_img.stream(process_image, [input_img, conf_threshold, mode], [input_img], stream_every=0.075)

if __name__ == "__main__":
    demo.launch(server_port=8051)
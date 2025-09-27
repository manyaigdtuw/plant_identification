# -*- coding: utf-8 -*-
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Configuration
img_size = 224
model_path = 'medicinal_leaf_model.h5'  # Path to your trained model
class_names_path = 'class_names.json'  # Path to class names file

# Load model and class names
def load_model_and_classes():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")
    
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file '{class_names_path}' not found.")
    
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading class names...")
    with open(class_names_path, 'r') as f:
        class_names_data = json.load(f)
    class_names = np.array(class_names_data['class_names'])
    
    return model, class_names

# Preprocessing function
def preprocess_image(img):
    img = img.convert("RGB")  
    img = img.resize((img_size, img_size))
    img_array = np.array(img).astype("float32")
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Prediction function
def predict(image):
    processed = preprocess_image(image)
    preds = model.predict(processed, verbose=0)[0]
    
    confidences = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    
    top_idx = np.argmax(preds)
    top_class = class_names[top_idx]
    top_confidence = preds[top_idx]
    
    result_text = f"Prediction: {top_class}\nConfidence: {top_confidence:.2%}"
    
    top5_indices = np.argsort(preds)[-5:][::-1]
    result_text += "\n\nTop 5 Predictions:"
    for i, idx in enumerate(top5_indices):
        result_text += f"\n{i+1}. {class_names[idx]}: {preds[idx]:.2%}"
    
    return result_text, confidences

def main():
    global model, class_names
    try:
        model, class_names = load_model_and_classes()
        print(f"Model loaded successfully with {len(class_names)} classes")
        print("Classes:", class_names)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create Gradio interface
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload a Leaf Image"),
        outputs=[
            gr.Textbox(label="Prediction Results", lines=5),
            gr.Label(label="All Class Probabilities", num_top_classes=10)
        ],
        title="Medicinal Leaf Classifier",
        description="Upload an image of a medicinal leaf and the AI model will predict its category.",
        examples=[
            ["example1.jpg"] if os.path.exists("example1.jpg") else None,
            ["example2.jpg"] if os.path.exists("example2.jpg") else None
        ]
    )
    
    # Launch the interface
    print("Launching Gradio interface...")
    print("The interface will open in your web browser at http://localhost:7860")
    demo.launch(share=False, server_name="0.0.0.0")  # Set share=True for public link

if __name__ == "__main__":
    main()
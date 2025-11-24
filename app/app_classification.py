import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import preprocess_image
import plotly.graph_objects as go
from ultralytics import YOLO
import cv2
import tempfile

# ----------------------------------------
# Page Config
# ----------------------------------------
st.set_page_config(
    page_title="Aerial Object Classifier - Bird vs Drone",
    page_icon="🚁",
    layout="wide"
)

st.title("🦅 Aerial Object Intelligence Dashboard")
st.markdown("### Choose between **Classification**, **YOLO Detection**, or **Live Webcam Mode**")

# ----------------------------------------
# Load Classification Model
# ----------------------------------------
import tensorflow as tf

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="app/model.tflite")
    interpreter.allocate_tensors()
    return interpreter

tflite_model = load_tflite_model()
def tflite_predict(img_array):
    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    tflite_model.set_tensor(input_index, img_array)
    tflite_model.invoke()

    return tflite_model.get_tensor(output_index)[0][0]


# ----------------------------------------
# Load YOLO Model
# ----------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("runs/detect/train5/weights/best.pt")

yolo_model = load_yolo_model()

# ----------------------------------------
# Grad-CAM Function
# ----------------------------------------
#def generate_gradcam(image, model_layer="mobilenetv2_1.00_224"):
    #model = tf.keras.models.load_model("models/final_cnn_model.keras", compile=False)

    #grad_model = tf.keras.models.Model(
    #    inputs=model.inputs,
     #   outputs=[model.get_layer(model_layer).output, model.output]
    #)

    #img_array = preprocess_image(image)
    #with tf.GradientTape() as tape:
     #   conv_outputs, predictions = grad_model(img_array)
      #  loss = predictions[:, 0]

    #grads = tape.gradient(loss, conv_outputs)[0]
    #weights = tf.reduce_mean(grads, axis=(0, 1))
    #cam = np.maximum(np.sum(weights * conv_outputs[0], axis=-1), 0)

    #cam = cv2.resize(cam, (image.width, image.height))
    #heatmap = cv2.applyColorMap(np.uint8(255 * cam / cam.max()), cv2.COLORMAP_JET)
    #heatmap = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

    #return heatmap

# ----------------------------------------
# Mode Selector
# ----------------------------------------
option = st.radio(
    "Select Mode:",
    ["Classification (Bird vs Drone)", "Object Detection (YOLO)", "Webcam Live Detection"],
    horizontal=True
)

# ----------------------------------------
# FILE UPLOAD MODE
# ----------------------------------------
uploaded_file = None
if option != "Webcam Live Detection":
    uploaded_file = st.file_uploader(
        "📸 Upload an Aerial Image",
        type=["jpg", "jpeg", "png"]
    )

if uploaded_file and option == "Classification (Bird vs Drone)":
    image_obj = Image.open(uploaded_file).convert("RGB")
    st.image(image_obj, caption="Uploaded Image")

    img_array = preprocess_image(image_obj)
    prediction = tflite_predict(img_array)


    label = "Drone 🚁" if prediction > 0.5 else "Bird 🐦"
    confidence = prediction if label == "Drone 🚁" else 1 - prediction

    st.subheader("🔮 Prediction Result")
    st.metric("Predicted Class", label)
    st.metric("Confidence", f"{confidence*100:.2f}%")

    # Grad-CAM
    #st.subheader("🔥 Grad-CAM Heatmap")
    #heatmap = generate_gradcam(image_obj)
    #st.image(heatmap, caption="Model Attention Map")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': f"Confidence Score ({label})"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# YOLO IMAGE DETECTION
# ----------------------------------------
elif uploaded_file and option == "Object Detection (YOLO)":
    image_obj = Image.open(uploaded_file).convert("RGB")
    st.image(image_obj, caption="Uploaded Image")

    with st.spinner("Running YOLO detection..."):
        results = yolo_model(image_obj, save=False)
        result_img = results[0].plot()

    st.subheader("📍 Detection Result")
    st.image(result_img, caption="YOLO Output")

    boxes = results[0].boxes.cls.tolist()
    st.write(f"✅ Birds detected: {boxes.count(0)}")
    st.write(f"✅ Drones detected: {boxes.count(1)}")

# ----------------------------------------
# ✅ WEBCAM LIVE YOLO MODE
# ----------------------------------------
elif option == "Webcam Live Detection":
    st.subheader("🎥 Live Drone/Bird Detection")

    run_cam = st.checkbox("Start Webcam")

    if run_cam:
        cap = cv2.VideoCapture(0)

        st_frame = st.empty()

        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Unable to access webcam!")
                break

            results = yolo_model(frame)
            annotated = results[0].plot()

            st_frame.image(annotated, channels="BGR")

        cap.release()

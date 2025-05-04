import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("asl_model_mobilenetv2_finetuned.h5")
class_names = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Streamlit UI setup
st.set_page_config(page_title="ISL Recognition", layout="centered", page_icon="🧠")

# Sidebar
with st.sidebar:
    st.title("🧠 ISL Recognition")
    st.markdown("Use your webcam to recognize **Indian Sign Language** gestures in real-time.")
    run = st.toggle("🎥 Start Webcam")
    st.markdown("---")
    st.info("Make sure your hand is visible in front of the camera and well-lit.")
    st.markdown("Developed with ❤️ using MediaPipe, MobileNetV2 & Streamlit.")

FRAME_WINDOW = st.empty()

# Start video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.6) as hands:

    while run:
        success, frame = cap.read()
        if not success:
            st.error("⚠️ Unable to access your webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Bounding box
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                xmin = int(min(x_coords) * w) - 20
                ymin = int(min(y_coords) * h) - 20
                xmax = int(max(x_coords) * w) + 20
                ymax = int(max(y_coords) * h) + 20

                xmin, ymin = max(xmin, 0), max(ymin, 0)
                xmax, ymax = min(xmax, w), min(ymax, h)

                hand_img = frame[ymin:ymax, xmin:xmax]

                if hand_img.size != 0:
                    try:
                        hand_img = cv2.resize(hand_img, (128, 128))
                        hand_img = hand_img.astype('float32') / 255.0
                        input_data = np.expand_dims(hand_img, axis=0)

                        prediction = model.predict(input_data, verbose=0)
                        predicted_letter = class_names[np.argmax(prediction)]

                        # Draw prediction
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (100, 255, 100), 2)
                        cv2.putText(frame, f"{predicted_letter}", (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 255, 50), 3)
                    except Exception as e:
                        st.warning(f"Prediction Error: {e}")
        else:
            cv2.putText(frame, "🖐 Show your hand clearly!", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Update frame in Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

# Stop the video
if not run:
    cap.release()
    st.success("✅ Webcam stopped.")

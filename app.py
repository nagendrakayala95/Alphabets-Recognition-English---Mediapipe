import streamlit as st
import cv2

from preprocessing import preprocess_image
from predict import predict_alphabet

st.set_page_config(page_title="Alphabet Recognition")

st.title("Alphabet Recognition")

run = st.checkbox("Start Camera")

frame_window = st.image([])

from streamlit_webrtc import webrtc_streamer

webrtc_streamer(
    key="camera",
    media_stream_constraints={"video": True, "audio": False},
)

while run:

    success, frame = camera.read()

    if not success:
        st.error("Cannot open camera")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    size = 250

    x1 = w // 2 - size // 2
    y1 = h // 2 - size // 2

    x2 = x1 + size
    y2 = y1 + size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    roi = frame[y1:y2, x1:x2]

    processed = preprocess_image(roi)

    prediction = predict_alphabet(processed)

    cv2.putText(
        frame,
        f"Prediction : {prediction}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()

import streamlit as st
import av
import cv2

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)

from preprocessing import preprocess_image
from predict import predict_alphabet

st.set_page_config(
    page_title="Alphabet Recognition",
    layout="centered"
)

st.title("🖐 Alphabet Recognition")
st.write("Show your hand sign inside the camera.")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        h, w, _ = img.shape

        size = 250

        x1 = w // 2 - size // 2
        y1 = h // 2 - size // 2

        x2 = x1 + size
        y2 = y1 + size

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        roi = img[y1:y2, x1:x2]

        try:

            processed = preprocess_image(roi)

            prediction = predict_alphabet(processed)

            cv2.putText(
                img,
                f"Prediction : {prediction}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        except Exception:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="alphabet",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)

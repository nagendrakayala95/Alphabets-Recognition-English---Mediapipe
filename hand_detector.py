import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.mp_draw = mp.solutions.drawing_utils

    def detect_hand(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hand_crop = None

        if results.multi_hand_landmarks:

            h, w, _ = frame.shape

            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                x_list = []
                y_list = []

                for lm in hand_landmarks.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

                xmin = max(min(x_list) - 20, 0)
                ymin = max(min(y_list) - 20, 0)
                xmax = min(max(x_list) + 20, w)
                ymax = min(max(y_list) + 20, h)

                hand_crop = frame[ymin:ymax, xmin:xmax]

        return frame, hand_crop
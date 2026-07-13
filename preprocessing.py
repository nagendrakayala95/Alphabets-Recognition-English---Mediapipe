import cv2
import numpy as np

IMAGE_SIZE = (34, 34)

def preprocess_image(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, -1)
    return img
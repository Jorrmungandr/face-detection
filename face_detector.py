import cv2
import numpy as np
from fastapi import HTTPException

HAAR_CASCADE_PATH = "haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def check_face_requirements(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="NO_FACE")

    x, y, w, h = faces[0]

    face_region = gray[y:y+h, x:x+w]
    avg_brightness = np.mean(face_region)

    if avg_brightness < 50:
        raise HTTPException(status_code=400, detail="LOW_BRIGHTNESS")
    elif avg_brightness > 205:
        raise HTTPException(status_code=400, detail="HIGH_BRIGHTNESS")

    frame_height, frame_width = gray.shape
    face_center_x = x + w/2
    face_center_y = y + h/2
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    x_threshold = 0.2 * frame_width
    y_threshold = 0.2 * frame_height

    dx = abs(face_center_x - frame_center_x)
    dy = abs(face_center_y - frame_center_y)

    if dx > x_threshold or dy > y_threshold:
        raise HTTPException(status_code=400, detail="FACE_DECENTRALIZED")

    return [x.item(), y.item(), w.item(), h.item()]

import cv2
import mediapipe
import pygame
import numpy
import sklearn
import PIL
import reportlab
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mediapipe.__version__)
print("Pygame:", pygame.__version__)
print("NumPy:", numpy.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Pillow:", PIL.__version__)
print("ReportLab: OK")

# Testa abertura da webcam
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("Webcam: OK — resolução:", frame.shape[1], "x", frame.shape[0])
    cap.release()
else:
    print("Webcam: ERRO — verifique /dev/video*")
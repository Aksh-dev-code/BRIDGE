import os 
import cv2
import mediapipe as np
import matplotlib.pyplot as plt

Data_DIR = '../data'

for dir_ in os.listdir(Data_DIR):
    for img_path in os.listdir(os.path.join)
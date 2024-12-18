import cv2
import mediapipe as mp
import torch
import numpy as np
import cvzone
from config import DEVICE

lm_list = []

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

def make_landmark_timestep(results):
    c_lm = []
    if results.pose_landmarks:
        print("LANDMARKS : ",results.pose_landmarks)
        for lm in results.pose_landmarks.landmark:
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
            c_lm.append(lm.presence)
    return c_lm

def draw_landmark_on_image(mp_draw, results, frame):
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame


def draw_bounding_box_and_label(frame, results, label, weapon_detected):
    """
    Draw bounding boxes and overlay colors based on detected actions and weapon presence.
    """
    if results.pose_landmarks:
        x_min, y_min = 1, 1
        x_max, y_max = 0, 0
        for lm in results.pose_landmarks.landmark:
            x_min = min(x_min, lm.x)
            y_min = min(y_min, lm.y)
            x_max = max(x_max, lm.x)
            y_max = max(y_max, lm.y)
        
        h, w, c = frame.shape
        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)

        status = None
        # Define overlay behavior based on label and weapon detection
        overlay = frame.copy()
        if label  == "Punch" and weapon_detected:  # Red overlay for punch/kick + weapon
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            status = "Danger"
            
        elif label == "Punch":  # Yellow overlay for punch/kick without weapon
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            status = "Warning"
        else:  # Neutral or no significant action
            color = (0, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Add the label text
        label_text = f"{label}"
        if weapon_detected:
            label_text += " + Weapon Detected"
        
        cvzone.putTextRect(
            frame, 
            label_text, 
            (max(0, x_min), max(40, y_min)), 
            scale=1, 
            thickness=1, 
            offset=5, 
            colorR=(0, 0, 0)
        )

    return frame , status


"""
Driver Drowsiness Detection System
===================================
Assignment: Detection Brainstorm - Solution Design
Uses face landmark detection + Eye Aspect Ratio (EAR) to detect drowsiness.

Requirements:
    pip install opencv-python dlib scipy numpy pygame imutils

Also download the dlib landmark model:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    Extract and place shape_predictor_68_face_landmarks.dat in the same folder.
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import time
import sys
import os

# ─── CONFIG ─────────────────────────────────────────────────────────────────
EAR_THRESHOLD   = 0.25   # Eye Aspect Ratio below this = eyes closing
CONSEC_FRAMES   = 20     # How many consecutive frames before alarm triggers
MODEL_PATH      = "shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
ALARM_SOUND     = "alarm.wav"   # Optional: place alarm.wav in the same folder
# ─────────────────────────────────────────────────────────────────────────────
def eye_aspect_ratio(eye):
    """
    Compute the Eye Aspect Ratio (EAR) for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    When the eye is open  → EAR ≈ 0.25–0.35
    When the eye is closed → EAR ≈ 0.0

    Args:
        eye: numpy array of 6 (x, y) landmark points

    Returns:
        float: EAR value
    """
    # Vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def draw_eye_contour(frame, eye_points, color=(0, 255, 0)):
    """Draw a green contour around the detected eye."""
    hull = cv2.convexHull(eye_points)
    cv2.drawContours(frame, [hull], -1, color, 1)


def play_alarm():
    """Play an alarm sound if alarm.wav exists, otherwise print a beep."""
    if os.path.exists(ALARM_SOUND):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(ALARM_SOUND)
                pygame.mixer.music.play(-1)   # loop
        except Exception as e:
            print(f"[Audio] Could not play alarm: {e}")
    else:
        print("\a", end="", flush=True)   # Terminal bell fallback


def stop_alarm():
    """Stop the alarm sound."""
    try:
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass


def load_detector_and_predictor(model_path):
    """Load dlib face detector and facial landmark predictor."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        sys.exit(1)

    print("[INFO] Loading dlib face detector and predictor...")
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    return detector, predictor


def get_eye_indices():
    """
    Return start/end indices for left and right eye landmarks
    in the 68-point dlib model.
    """
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    return (lStart, lEnd), (rStart, rEnd)


def draw_hud(frame, ear, counter, alert_active, fps):
    """Draw the heads-up display overlay on the frame."""
    h, w = frame.shape[:2]

    # Background bar at top
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

    # EAR value
    ear_color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255)
    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_color, 2)

    # Frame counter
    cv2.putText(frame, f"Frames: {counter}/{CONSEC_FRAMES}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Drowsy alert banner
    if alert_active:
        cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, "DROWSINESS DETECTED! WAKE UP!", (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)


def run_detection():
    """Main loop: capture video, detect drowsiness, show alerts."""
    detector, predictor = load_detector_and_predictor(MODEL_PATH)
    (lStart, lEnd), (rStart, rEnd) = get_eye_indices()

    print("[INFO] Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    frame_counter  = 0   # consecutive low-EAR frames
    alert_active   = False
    prev_time      = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Lost webcam feed.")
            break

        # ── FPS calculation ─────────────────────────────────────────────────
        now  = time.time()
        fps  = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        # ── Pre-process frame ────────────────────────────────────────────────
        frame = cv2.flip(frame, 1)          # mirror for natural feel
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Face detection ───────────────────────────────────────────────────
        faces = detector(gray, 0)
        ear   = 1.0   # default to "open" if no face found

        for face in faces:
            # Draw face bounding box
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 1)

            # Get 68 facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Extract eye landmark coordinates
            leftEye  = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Compute EAR for both eyes, average them
            leftEAR  = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear      = (leftEAR + rightEAR) / 2.0

            # Draw eye contours
            eye_color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255)
            draw_eye_contour(frame, leftEye,  eye_color)
            draw_eye_contour(frame, rightEye, eye_color)

            # ── Drowsiness logic ─────────────────────────────────────────────
            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSEC_FRAMES:
                    if not alert_active:
                        alert_active = True
                        play_alarm()
            else:
                frame_counter = 0
                if alert_active:
                    alert_active = False
                    stop_alarm()

        # ── Draw HUD ─────────────────────────────────────────────────────────
        draw_hud(frame, ear, frame_counter, alert_active, fps)

        cv2.imshow("Drowsiness Detection — Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("[INFO] Shutting down...")
    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_detection()
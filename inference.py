import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import threading
import os
import time
import json

# --- Camera capture cap initialization---
class WebcamStream:
    def __init__(self, src=0):
        # For native camera
        if isinstance(src, int):
            self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.stream = cv2.VideoCapture(src)

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.stopped = False
        self.lock = threading.Lock()

        # --- HARDWARE WARMUP DELAY ---
        self.grabbed = False
        self.frame = None
        for _ in range(30):
            grabbed, frame = self.stream.read()
            if grabbed and frame is not None:
                self.grabbed = True
                self.frame = frame
                break
            time.sleep(0.1)

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        consecutive_failures = 0
        max_failures = 30  # 30 consecutive failed frames result to stopping
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self.stop()
                    break
                time.sleep(0.01)
                continue

            consecutive_failures = 0
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame

    def isOpened(self):
        with self.lock:
            return self.stream.isOpened() and self.grabbed

    def set(self, propId, value):
        return self.stream.set(propId, value)

    def get(self, propId):
        return self.stream.get(propId)

    def stop(self):
        self.stopped = True

    def release(self):
        self.stop()
        self.stream.release()

# ---  The Neural Network Model ---
class SignLanguageLSTM(nn.Module):
    def __init__(self, num_classes, input_size=195, hidden_size=128, num_layers=2):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_thought = lstm_out[:, -1, :]
        return self.fc(final_thought)


def extract_and_normalize(landmarks, nose_x, nose_y, nose_z, is_pose=False):
    if not landmarks:
        return np.zeros(69 if is_pose else 63).tolist()

    target_landmarks = landmarks.landmark[:23] if is_pose else landmarks.landmark
    raw_coords = []

    for lm in target_landmarks:
        raw_coords.extend([lm.x - nose_x, lm.y - nose_y, lm.z - nose_z])

    max_val = max(list(map(abs, raw_coords))) if raw_coords else 0
    return [c / max_val for c in raw_coords] if max_val > 0 else raw_coords


def enforce_4_3_aspect_ratio(frame):
    if frame is None:
        return None
    h, w, _ = frame.shape
    target_aspect = 4 / 3
    current_aspect = w / h

    # Epsilon tolerance to avoid floating-point comparison issues
    if abs(current_aspect - target_aspect) < 1e-4:
        return frame

    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        start_x = (w - new_w) // 2
        return frame[:, start_x:start_x + new_w]
    else:
        new_h = int(w / target_aspect)
        start_y = (h - new_h) // 2
        return frame[start_y:start_y + new_h, :]


def main():
    # --- Load Class Mapping ---
    try:
        with open('classes.json', 'r') as f:
            CLASSES = json.load(f)
        print(f"Loaded classes: {CLASSES}")
    except FileNotFoundError:
        print("Error: classes.json not found. Please train the model first.")
        return

    # ---  Configuration ---
    SEQUENCE_LENGTH = 45

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model and load trained weights
    model = SignLanguageLSTM(num_classes=len(CLASSES))
    model_path = 'sign_language_model_test(Sep13).pth'

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Successfully loaded model weights from '{model_path}'")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("This usually happens if the number of classes in the saved weights does not match len(CLASSES).")
            print("Running with uninitialized weights for visualization/debugging.")
    else:
        print(f"Warning: Model weight file '{model_path}' not found.")
        print("Running with uninitialized weights for visualization/debugging.")

    model.to(device)
    model.eval()

    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize the "Rolling Window" conveyer belt buffer
    sequence = deque(maxlen=SEQUENCE_LENGTH)


    # --- Camera initialization ---
    cap = WebcamStream(src=1).start()
    if not cap.isOpened():
        print("Warning: Camera index 1 not found. Falling back to index 0...")
        cap.release()
        # Re-initialize the class with the fallback index
        cap = WebcamStream(src=0).start()

        if not cap.isOpened():
            print("Error: No functional camera detected. Exiting.")
            return


    # --- The Real-Time Game Loop ---
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break
            frame = enforce_4_3_aspect_ratio(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB for MediaPipe
            results = holistic.process(frame_rgb)

            prediction_text = "No Pose Detected"
            # --- DRAW LANDMARKS ---
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            display_frame = cv2.flip(frame, 1)

            # --- NOSE-CENTRIC NORMALIZATION ---
            nose_x, nose_y, nose_z = 0.0, 0.0, 0.0
            if results.pose_landmarks:
                nose_x = results.pose_landmarks.landmark[0].x
                nose_y = results.pose_landmarks.landmark[0].y
                nose_z = results.pose_landmarks.landmark[0].z

            pose_data = extract_and_normalize(results.pose_landmarks, nose_x, nose_y, nose_z, is_pose=True)
            left_hand_data = extract_and_normalize(results.left_hand_landmarks, nose_x, nose_y, nose_z)
            right_hand_data = extract_and_normalize(results.right_hand_landmarks, nose_x, nose_y, nose_z)

            full_frame_data = pose_data + left_hand_data + right_hand_data
            sequence.append(full_frame_data)

            # --- The Prediction Engine ---
            if len(sequence) < SEQUENCE_LENGTH:
                prediction_text = f"Warming up ({len(sequence)}/{SEQUENCE_LENGTH})..."
            elif np.sum(np.abs(sequence[-1][:69])) == 0.0:  # Check if the most recent frame has body (69 features)
                prediction_text = "Waiting for person..."
            else:
                # Format the data for PyTorch (1 batch, 45 frames, 195 features)
                input_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    confidence_pct = confidence.item() * 100

                    if confidence_pct > 75.0:
                        current_guess = CLASSES[predicted_idx.item()]
                        prediction_text = f"{current_guess} - {confidence_pct:.1f}%"
                    else:
                        prediction_text = "Thinking..."

            # --- Render User Interface ---
            cv2.putText(display_frame, prediction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(display_frame, f"Buffer: {len(sequence)}/{SEQUENCE_LENGTH}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "press q to exit", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 200, 0), 2, cv2.LINE_AA)
            cv2.imshow('LSTM Sequence Recognition', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('LSTM Sequence Recognition', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading

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
        # Try to read a valid frame with retries to allow camera sensor warmup
        self.grabbed = False
        self.frame = None
        for _ in range(30):  # Up to 3.0 seconds warmup (30 attempts * 0.1s sleep)
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
        max_failures = 30  # Allow up to 30 consecutive failed frames before stopping
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self.stop()
                    break
                time.sleep(0.01)  # Short sleep to avoid tight loop during failure
                continue

            consecutive_failures = 0
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame

    def isOpened(self):
        # Secure the boolean read with the mutex lock
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

    # Use an epsilon tolerance to avoid floating-point comparison issues
    if abs(current_aspect - target_aspect) < 1e-4:
        return frame

    if current_aspect > target_aspect:
        # Too wide (e.g., 16:9) -> Crop width
        new_w = int(h * target_aspect)
        start_x = (w - new_w) // 2
        return frame[:, start_x:start_x + new_w]
    else:
        # Too tall (e.g., 9:16) -> Crop height
        new_h = int(w / target_aspect)
        start_y = (h - new_h) // 2
        return frame[start_y:start_y + new_h, :]


def main():
    # --- Configuration ---
    DATA_PATH = 'extracted_data'
    SEQUENCE_LENGTH = 45  # 1.5 seconds to capture complex signs like 'J and Z'
    SEQUENCES_TO_COLLECT = 30  # Number of videos to record per session

    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        model_complexity=0,  # 0 = Lite (fastest, prevents lag), 1 = Full, 2 = Heavy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    # Setup the folders
    action_name = input("Enter the word you want to record: ").strip().upper()
    contributor_id = input("Enter your initials (e.g., JM): ").strip().upper()
    date_id = input("Enter mode of capture (e.g., p(for phone) or n(for native) camera: ").strip().upper()
    action_path = os.path.join(DATA_PATH, action_name)
    os.makedirs(action_path, exist_ok=True)


    # ---Camera initialization---
    cap = WebcamStream(src=1).start()
    if not cap.isOpened():
        print("Warning: Camera index 1 not found. Falling back to index 0...")
        cap.release()
        # Re-initialize the class with the fallback index
        cap = WebcamStream(src=0).start()

        if not cap.isOpened():
            print("Error: No functional camera detected. Exiting.")
            return
    #-------------------------

    print(f"\n--- Starting Data Collection for '{action_name}' ---")
    print("Press 'q' at any time to quit early.")

    # --- DOUBLE CHECK CAMERA SETTINGS ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hardware_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera hardware negotiated at {actual_height}p at {hardware_fps} FPS limit.")
    print(f"--------------------------")

    # The Main Recording Loop
    try:
        sequence_num = 0
        while sequence_num < SEQUENCES_TO_COLLECT:
            # --- PHASE A: The Rest/Countdown Phase ---
            start_time = time.time()
            while time.time() - start_time < 1.0:  # break between recordings
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = enforce_4_3_aspect_ratio(frame)
                frame = cv2.flip(frame, 1)
                time_left = 1.1 - (time.time() - start_time)
                cv2.putText(frame, f"GET READY: {time_left:.1f}s", (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.putText(frame, f"Recording Sequence {sequence_num + 1}/{SEQUENCES_TO_COLLECT}",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2, cv2.LINE_AA)
                cv2.imshow('Data Collection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        
            # --- PHASE B: The Recording Phase ---
            sequence_data = []
            prev_frame_time = time.time()
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = enforce_4_3_aspect_ratio(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)

                # Draw Landmarks (Disabled to prioritize FPS)
                '''
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                #'''
                # ------------------

                display_frame = cv2.flip(frame, 1)

                # --- NOSE-CENTRIC NORMALIZATION ---
                # Default to 0,0,0 if nose isn't detected
                nose_x, nose_y, nose_z = 0.0, 0.0, 0.0
                if results.pose_landmarks:
                    nose_x = results.pose_landmarks.landmark[0].x
                    nose_y = results.pose_landmarks.landmark[0].y
                    nose_z = results.pose_landmarks.landmark[0].z

                # Extract features
                pose_data = extract_and_normalize(results.pose_landmarks, nose_x, nose_y, nose_z, is_pose=True)
                left_hand_data = extract_and_normalize(results.left_hand_landmarks, nose_x, nose_y, nose_z)
                right_hand_data = extract_and_normalize(results.right_hand_landmarks, nose_x, nose_y, nose_z)

                # Combine into 195-feature array
                full_frame_data = pose_data + left_hand_data + right_hand_data
                sequence_data.append(full_frame_data)

                # Calculate true FPS
                new_frame_time = time.time()
                true_fps = int(1 / (new_frame_time - prev_frame_time))
                prev_frame_time = new_frame_time

                # UI updates
                cv2.putText(display_frame, f'FPS: {true_fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Data Collection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # --- PHASE C: Save the File ---
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = enforce_4_3_aspect_ratio(frame)
                frame = cv2.flip(frame, 1)

                # Show validation instructions
                cv2.putText(frame, f"Seq {sequence_num + 1} Captured!", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, "Press 'Spacebar' to NEXT (Save) | 'r' to RETRY", (15, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow('Data Collection', frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 32: # "ord('n')"-> use this if other keys are preferred
                    if len(sequence_data) != SEQUENCE_LENGTH:
                        print(
                            f"\n[ERROR] Sequence shape mismatch! Captured {len(sequence_data)} frames, expected {SEQUENCE_LENGTH}.")
                        print("This sequence cannot be saved. Please press 'r' to retry.")
                        continue

                    # Save the File and advance
                    np_data = np.array(sequence_data)
                    timestamp = int(time.time() * 1000)
                    file_path = os.path.join(action_path, f"{action_name}_{contributor_id}_{timestamp}_{date_id}.npy")
                    np.save(file_path, np_data)
                    print(f"Saved: {file_path} | Shape: {np_data.shape}")

                    sequence_num += 1  # Advance to the next sequence
                    break

                elif key == ord('r'):
                    print(f"Retrying sequence {sequence_num + 1}...")
                    break

                elif key == ord('q') or cv2.getWindowProperty('Data Collection', cv2.WND_PROP_VISIBLE) < 1:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
    finally:
        print("\n--- Collection Complete! ---")
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

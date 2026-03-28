import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

VIDEO_DIR = 'dataset_videos'
OUTPUT_DIR = 'extracted_data'
SEQUENCE_LENGTH = 45

os.makedirs(OUTPUT_DIR, exist_ok=True)

for action in os.listdir(VIDEO_DIR):
    action_path = os.path.join(VIDEO_DIR, action)
    if not os.path.isdir(action_path):
        continue

    action_output_dir = os.path.join(OUTPUT_DIR, action)
    os.makedirs(action_output_dir, exist_ok=True)

    for video_name in os.listdir(action_path):
        video_path = os.path.join(action_path, video_name)
        cap = cv2.VideoCapture(video_path)
        all_frames_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- START OF ADAPTIVE ILLUMINATION FILTER ---
            # Convert the frame to grayscale just to measure the light intensity
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Calculate average pixel brightness (0 is pitch black, 255 is pure white)
            average_brightness = cv2.mean(gray_frame)[0]

            # The Conditional Gate: Only boost if the frame is too dark (e.g., below 80)
            if average_brightness < 80:
                # alpha = 1.2 (boosts contrast by 20% to keep finger edges sharp)
                # beta = 30 (adds a flat +30 brightness to every pixel)
                # convertScaleAbs safely bounds the math so pixels don't exceed pure white (255)
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # --- THE NORMALIZATION BLOCK ---
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    base_z = hand_landmarks.landmark[0].z

                    raw_coords = []
                    for lm in hand_landmarks.landmark:
                        norm_x = lm.x - base_x
                        norm_y = lm.y - base_y
                        norm_z = lm.z - base_z
                        raw_coords.extend([norm_x, norm_y, norm_z])

                    max_value = max(list(map(abs, raw_coords)))
                    if max_value > 0:
                        normalized_coords = [c / max_value for c in raw_coords]
                    else:
                        normalized_coords = raw_coords

                    all_frames_data.append(normalized_coords)

            else:
                all_frames_data.append(np.zeros(63).tolist())

        cap.release()

        # --- THE ROUTING LOGIC ---
        # Data Augmentation for Static Signs
        num_sequences = len(all_frames_data) // SEQUENCE_LENGTH

        for i in range(num_sequences):
            start_idx = i * SEQUENCE_LENGTH
            end_idx = start_idx + SEQUENCE_LENGTH
            sequence_data = all_frames_data[start_idx:end_idx]

            np_data = np.array(sequence_data)
            output_file = os.path.join(action_output_dir, f"{video_name.split('.')[0]}_slice{i}.npy")
            np.save(output_file, np_data)
            print(f"Saved Static Slice: {output_file} | Shape: {np_data.shape}")

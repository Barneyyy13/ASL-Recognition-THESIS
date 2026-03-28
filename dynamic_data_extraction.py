import cv2
import mediapipe as mp
import numpy as np
import os
import time


def main():
    # --- Configuration ---
    DATA_PATH = 'extracted_data'
    SEQUENCE_LENGTH = 45  # 1.5 seconds to capture complex signs like 'J and Z'
    SEQUENCES_TO_COLLECT = 15  # Number of videos to record per session

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,  # Helps find the hand initially
        min_tracking_confidence=0.5  # Helps hold onto the hand during fast sweeps
    )

    # Setup the folders
    action_name = input("Enter the letter you want to record (e.g., J, Z): ").strip().upper()
    action_path = os.path.join(DATA_PATH, action_name)
    os.makedirs(action_path, exist_ok=True)

    # Checks how many files already exist to avoid overwriting of old data
    existing_files = os.listdir(action_path)
    start_idx = len(existing_files)

    cap = cv2.VideoCapture(0)
    print(f"\n--- Starting Data Collection for '{action_name}' ---")
    print("Press 'q' at any time to quit early.")

    # The Main Recording Loop
    for sequence_num in range(start_idx, start_idx + SEQUENCES_TO_COLLECT):

        # --- PHASE A: The Rest/Countdown Phase ---
        start_time = time.time()
        while time.time() - start_time < 1.5:  # 1.5-second break between recordings
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            time_left = 2.0 - (time.time() - start_time)
            cv2.putText(frame, f"GET READY: {time_left:.1f}s", (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, f"Recording Sequence {sequence_num + 1}/{start_idx + SEQUENCES_TO_COLLECT}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2, cv2.LINE_AA)

            cv2.imshow('Data Collection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # --- PHASE B: The Recording Phase ---
        sequence_data = []
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # --- THE NORMALIZATION BLOCK ---
                    #  Grab the Wrist (Landmark 0) as our absolute Origin
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    base_z = hand_landmarks.landmark[0].z

                    raw_coords = []
                    for lm in hand_landmarks.landmark:
                        #  Translation (Subtract wrist from every joint)
                        norm_x = lm.x - base_x
                        norm_y = lm.y - base_y
                        norm_z = lm.z - base_z
                        raw_coords.extend([norm_x, norm_y, norm_z])

                    max_value = max(list(map(abs, raw_coords)))

                    if max_value > 0:
                        normalized_coords = [c / max_value for c in raw_coords]
                    else:
                        normalized_coords = raw_coords

                    sequence_data.append(normalized_coords)

            else:
                # If hand is completely lost, pad with zeros
                sequence_data.append(np.zeros(63).tolist())

            cv2.putText(frame, f"RECORDING...", (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.imshow('Data Collection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # --- PHASE C: Save the File ---
        np_data = np.array(sequence_data)
        file_path = os.path.join(action_path, f"{action_name}_{sequence_num}.npy")
        np.save(file_path, np_data)
        print(f"Saved: {file_path} | Shape: {np_data.shape}")

    print("\n--- Collection Complete! ---")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
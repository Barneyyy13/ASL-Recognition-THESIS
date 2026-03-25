import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# --- 1. The Neural Network Model ---
class SignLanguageLSTM(nn.Module):
    def __init__(self, num_classes, input_size=63, hidden_size=128, num_layers=2):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_thought = lstm_out[:, -1, :]
        return self.fc(final_thought)


def main():
    # --- 2. Configuration ---
    # UPDATE THIS array to match the exact folder names in your dataset
    CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'IDLE', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']
    SEQUENCE_LENGTH = 45  # 1.5 seconds window for complex signs like 'J and Z'

    # To initialize the model and load trained weights
    model = SignLanguageLSTM(num_classes=len(CLASSES))
    model.load_state_dict(torch.load('sign_language_model.pth'))
    model.eval()

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,  # Optimization to hold onto blurry motion
        min_tracking_confidence=0.5
    )

    # Initialize the "Rolling Window" conveyer belt buffer
    sequence = deque(maxlen=SEQUENCE_LENGTH)

    # Open the Webcam
    cap = cv2.VideoCapture(0)
    print("Webcam opened. Press 'q' to quit.")

    # --- 3. The Real-Time Game Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB for MediaPipe
        results = hands.process(frame_rgb)

        prediction_text = "No Hand Detected, Waiting for motion..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ==========================================
                # --- THE NORMALIZATION BLOCK ---

                # Step A: Grab the Wrist (Landmark 0) as our absolute Origin (0,0,0)
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                base_z = hand_landmarks.landmark[0].z

                raw_coords = []
                for lm in hand_landmarks.landmark:
                    # Step B: Translation (Subtract wrist coordinates from every joint)
                    norm_x = lm.x - base_x
                    norm_y = lm.y - base_y
                    norm_z = lm.z - base_z
                    raw_coords.extend([norm_x, norm_y, norm_z])

                # Step C: Scaling (Find the max distance to use as a divider)
                max_value = max(list(map(abs, raw_coords)))

                # Step D: Divide everything by max_value to squash between -1.0 and 1.0
                if max_value > 0:
                    normalized_coords = [c / max_value for c in raw_coords]
                else:
                    normalized_coords = raw_coords

                # Append the final, normalized math to the live buffer
                sequence.append(normalized_coords)

                # ==========================================

        else:
            # If hand is lost, pad with zeros to reset the model's memory
            sequence.append(np.zeros(63).tolist())

        # --- 4. The Prediction Engine ---
        if len(sequence) == SEQUENCE_LENGTH:

            if np.sum(np.abs(sequence[-1])) == 0.0:  # Check if the most recent frame has no hand
                prediction_text = "Waiting for hand..."
            else:
                # Format the data for PyTorch (1 batch, 45 frames, 63 features)
                input_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(input_tensor)  # These are the raw logits

                    # Apply Softmax to convert logits to probabilities (0.0 to 1.0)
                    probabilities = F.softmax(outputs, dim=1)

                    # Get the highest probability and its corresponding index
                    confidence, predicted_idx = torch.max(probabilities, 1)

                    # Convert the decimal to a percentage (e.g., 0.985 -> 98.5)
                    confidence_pct = confidence.item() * 100

                    # Only display the guess if the model is highly confident
                    if confidence_pct > 75.0:
                        prediction_text = f"{CLASSES[predicted_idx.item()]} - {confidence_pct:.1f}%"
                    else:
                        prediction_text = "Thinking..."

        # --- 5. Render the User Interface ---
        cv2.putText(frame, prediction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Buffer: {len(sequence)}/{SEQUENCE_LENGTH}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "press q to exit", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 200, 0), 2, cv2.LINE_AA)
        cv2.imshow('LSTM Sequence Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
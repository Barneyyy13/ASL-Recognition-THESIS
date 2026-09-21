import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
import json

# --- 1. The PyTorch Dataset ---
class SignLanguageSequenceDataset(Dataset):
    def __init__(self, data_dir):
        # 1. ADDED self.filepaths
        self.data, self.labels, self.filepaths = [], [], []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for action in self.classes:
            action_path = os.path.join(data_dir, action)
            npy_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]

            if len(npy_files) == 0:
                print(f"[WARNING] Folder '{action}' is completely empty! The model will not learn this class.")
                continue

            for file in npy_files:
                file_path = os.path.join(action_path, file)
                try:
                    array = np.load(file_path)

                    if array.shape == (45, 195):
                        self.data.append(array)
                        self.labels.append(self.class_to_idx[action])
                        # 2. ADDED tracking of the path
                        self.filepaths.append(file_path)
                    else:
                        print(f"[WARNING] Skipping {file_path}: Invalid shape {array.shape}, expected (45, 195).")
                except Exception as e:
                    print(f"[WARNING] Skipping corrupted file {file_path}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load your existing 195-feature array
        sequence = self.data[idx].copy()  # Use .copy() to avoid altering the stored data in RAM

        # --- THE FIX: Spatial Dropout for Lower Pose ---
        # Zeroes out elbows, wrists, and pose-hands (indices 39 to 68)
        # Keeps Head/Shoulders (0-38) and the actual MediaPipe Hands (69-194)
        sequence[:, 39:69] = 0.0

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# --- 2. The Neural Network Model ---
class SignLanguageLSTM(nn.Module):
    def __init__(self,  num_classes, input_size=195, hidden_size=128, num_layers=2):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_thought = lstm_out[:, -1, :]
        return self.fc(final_thought)


# --- 3. The Training Loop ---
def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        dataset = SignLanguageSequenceDataset('extracted_data')
    except FileNotFoundError as e:
        print(e)
        return

    if len(dataset) == 0:
        print("Error: Dataset is empty. Please extract data before training.")
        return

        # --- Export Class Mapping for Inference ---
    with open('classes.json', 'w') as f:
        json.dump(dataset.classes, f)
    print(f"Exported class mapping to classes.json: {dataset.classes}")

    # --- PREVENT DATA LEAKAGE: Group augmented files with their raw parents ---
    base_files = {}  # Dictionary mapping base_name -> list of original dataset indices
    base_labels = {}  # Dictionary mapping base_name -> integer label

    # NOTE: Check your SignLanguageSequenceDataset class. It must have attributes
    # for the paths and labels (e.g., self.filepaths and self.labels).
    # Adjust 'dataset.filepaths' and 'dataset.labels' below to match your class variables.
    for idx, path in enumerate(dataset.filepaths):
        filename = os.path.basename(path)
        # Strip synthetic tags to find the root physical capture name
        base_name = filename.replace('_mirror', '') \
            .replace('_noise', '') \
            .replace('_fast', '') \
            .replace('_slow', '')

        if base_name not in base_files:
            base_files[base_name] = []
            base_labels[base_name] = dataset.labels[idx]

        base_files[base_name].append(idx)

    unique_bases = list(base_files.keys())
    unique_labels = [base_labels[b] for b in unique_bases]

    # Split the unique physical captures 80/20
    train_bases, val_bases = train_test_split(
        unique_bases,
        test_size=0.2,
        random_state=42,
        stratify=unique_labels
    )

    # Reconstruct the indices for the PyTorch DataLoader
    train_indices = []
    for base in train_bases:
        train_indices.extend(base_files[base])

    val_indices = []
    for base in val_bases:
        val_indices.extend(base_files[base])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Data Leakage Prevented: {len(train_bases)} unique captures in Train, {len(val_bases)} in Val.")
    # --------------------------------------------------------------------------

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SignLanguageLSTM(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- EARLY STOPPING SETUP ---
    best_val_loss = float('inf')  # Start with an infinitely high loss
    patience = 15  # How many epochs to wait before giving up
    patience_counter = 0  # Tracks how many epochs have failed to improve
    epochs = 500  # change back to 500 for final model
    loss_history = [] # used to visual plot
    for epoch in range(epochs):
        running_train_loss = 0.0
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        loss_history.append((avg_train_loss, avg_val_loss))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        #  STRICT VALIDATION EARLY STOPPING
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'sign_language_model_test_Validation(Sep21).pth')
            print(f"  -> Model improved! Saved with Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at Epoch {epoch + 1}! Training halted to prevent overfitting.")
                break

    #To use during final model training
    """"# --- GENERATE TRAINING GRAPH ---
    print("\nGenerating training performance graph...")

    # 1. Create a blank 10x6 inch canvas
    plt.figure(figsize=(10, 6))
    
    # 2. Unpack the train and validation losses from the history tuples
    train_losses = [x[0] for x in loss_history]
    val_losses = [x[1] for x in loss_history]
    
    # 3. Plot both curves
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2, linestyle='--')
    
    # 4. Add professional academic labels
    plt.title('LSTM Model Training Performance (Loss Curves)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Lower is Better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 5. Save the image to your project folder
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    print("Graph saved successfully as 'training_curve.png'!")
"""

if __name__ == "__main__":
    main()

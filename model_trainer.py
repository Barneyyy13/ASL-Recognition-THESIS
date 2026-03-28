import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# --- 1. The PyTorch Dataset ---
class SignLanguageSequenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data, self.labels = [], []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for action in self.classes:
            action_path = os.path.join(data_dir, action)
            for file in os.listdir(action_path):
                if file.endswith('.npy'):
                    self.data.append(np.load(os.path.join(action_path, file)))
                    self.labels.append(self.class_to_idx[action])

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# --- 2. The Neural Network Model ---
class SignLanguageLSTM(nn.Module):
    def __init__(self,  num_classes, input_size=63, hidden_size=128, num_layers=2):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_thought = lstm_out[:, -1, :]
        return self.fc(final_thought)


# --- 3. The Training Loop ---
def main():
    dataset = SignLanguageSequenceDataset('extracted_data')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SignLanguageLSTM(num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- EARLY STOPPING SETUP ---
    best_loss = float('inf')  # Start with an infinitely high loss
    patience = 15  # How many epochs to wait before giving up
    patience_counter = 0  # Tracks how many epochs have failed to improve
    loss_history = []
    epochs = 500

    for epoch in range(epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # --- EARLY STOPPING LOGIC ---
        # Calculate the average loss for this specific epoch
        current_loss = running_loss / len(train_loader)
        loss_history.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0  # Reset the strike counter because it improved
            torch.save(model.state_dict(), 'sign_language_model.pth')
            print(f"  -> Model improved! Saved with loss: {best_loss:.4f}")

        else:
            patience_counter += 1  # The model didn't improve, add a strike

            if patience_counter >= patience:
                print(f"Early stopping triggered at Epoch {epoch + 1}! Training halted safely.")
                break  # Instantly breaks out of the loop and stops training

    # --- GENERATE TRAINING GRAPH ---
    print("\nGenerating training performance graph...")

    # 1. Create a blank 10x6 inch canvas
    plt.figure(figsize=(10, 6))

    # 2. Plot the list of scores we tracked
    plt.plot(loss_history, label='Training Loss', color='blue', linewidth=2)

    # 3. Add professional academic labels
    plt.title('LSTM Model Training Performance (Loss Curve)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Lower is Better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. Save the image to your project folder
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    print("Graph saved successfully as 'training_curve.png'!")


if __name__ == "__main__":
    main()

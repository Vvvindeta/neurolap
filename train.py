import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
from neuralop.models.fno import FNO2d


class FNOImageClassifier(nn.Module):
    def __init__(self, n_modes_height=16, n_modes_width=16, hidden_channels=32, num_classes=3):
        super(FNOImageClassifier, self).__init__()
        self.fno = FNO2d(
            in_channels=3,
            out_channels=hidden_channels,
            n_modes_height=n_modes_height,
            n_modes_width=n_modes_width,
            hidden_channels=hidden_channels
        )
        self.fc = nn.Linear(hidden_channels * 256 * 256, num_classes)

    def forward(self, x):
        x = self.fno(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_model(dataset_path, model_name, model_params, epochs, batch_size, learning_rate):
    model_dir = "made_model"
    log_dir = "train_log"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(dataset.classes)
    model = FNOImageClassifier(
        n_modes_height=model_params["n_modes_height"],
        n_modes_width=model_params["n_modes_width"],
        hidden_channels=model_params["hidden_channels"],
        num_classes=num_classes
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Starting training...")
    start_time = time.time()
    log_file_name = os.path.join(log_dir, f"train_log_{model_name}.txt")
    with open(log_file_name, "a") as log_file:
        log_file.write(f"Training started for model: {model_name}\n")
        log_file.write(f"Parameters: {model_params}\n")
        log_file.write(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}\n")
        log_file.write("=" * 50 + "\n")
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader, 1):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                log_file.write(f"Epoch {epoch}/{epochs}, Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}\n")
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
            log_file.write(f"Epoch {epoch}/{epochs} completed, Average Loss: {epoch_loss:.4f}\n")
        log_file.write("Training completed.\n")
        log_file.write("=" * 50 + "\n")
    end_time = time.time()
    training_time = end_time - start_time
    print("Training complete. Saving model...")
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    models_file_path = os.path.join(model_dir, "models.txt")
    with open(models_file_path, "a") as file:
        file.write(f"Model Name: {model_name}\n")
        file.write(f"Parameters: {model_params}\n")
        file.write(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}\n")
        file.write(f"Training Time: {training_time:.2f} seconds\n")
        file.write(f"Model Path: {model_path}\n")
        file.write("=" * 50 + "\n")
    print(f"Model saved in {model_path}. Training details saved in {models_file_path} and {log_file_name}.")


if __name__ == "__main__":
    dataset_path = "images/dataset (2k)"
    model_name = "model5"
    model_params = {
        "n_modes_height": 16,
        "n_modes_width": 16,
        "hidden_channels": 64
    }
    epochs = 3
    batch_size = 8
    learning_rate = 0.001
    train_model(dataset_path, model_name, model_params, epochs, batch_size, learning_rate)

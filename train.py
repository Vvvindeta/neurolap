import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from neuralop.models.fno import FNO2d


class FNOImageClassifier(nn.Module):
    def __init__(self, n_modes_height=16, n_modes_width=16, hidden_channels=32, num_classes=4):
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


def train_model(data_path, model_name, n_modes_height, n_modes_width, hidden_channels, batch_size, lr, epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    num_classes = len(dataset.classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FNOImageClassifier(n_modes_height, n_modes_width, hidden_channels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    # Prepare log file
    os.makedirs("train_log", exist_ok=True)
    log_file_path = os.path.join("train_log", f"train_log_{model_name}.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Training started for model: {model_name}\n")
        log_file.write(f"Parameters: {{'n_modes_height': {n_modes_height}, 'n_modes_width': {n_modes_width}, 'hidden_channels': {hidden_channels}}}\n")
        log_file.write(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}\n")
        log_file.write("=" * 50 + "\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_index = 0
        for inputs, labels in dataloader:
            batch_index += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Log batch details
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Epoch {epoch + 1}/{epochs}, Batch {batch_index}/{len(dataloader)}, Loss: {loss.item():.4f}\n")

        avg_loss = epoch_loss / len(dataloader)
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Epoch {epoch + 1}/{epochs} completed, Average Loss: {avg_loss:.4f}\n")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"made_model/{model_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    training_time = time.time() - start_time

    # Finalize log file
    with open(log_file_path, "a") as log_file:
        log_file.write("Training completed.\n")
        log_file.write("=" * 50 + "\n")

    # Save model information to models.txt
    model_info = {
        "Model Name": model_name,
        "Parameters": {
            "n_modes_height": n_modes_height,
            "n_modes_width": n_modes_width,
            "hidden_channels": hidden_channels,
        },
        "Epochs": epochs,
        "Batch Size": batch_size,
        "Learning Rate": lr,
        "Training Time (seconds)": round(training_time, 2),
        "Final Loss": round(best_loss, 4),
        "Classes": dataset.classes,
        "Dataset Size": len(dataset),
        "Model Path": f"made_model/{model_name}.pth"
    }

    os.makedirs("made_model", exist_ok=True)
    models_file = os.path.join("made_model", "models.txt")

    with open(models_file, "a") as f:
        f.write(f"Model Name: {model_info['Model Name']}\n")
        f.write(f"Parameters: {model_info['Parameters']}\n")
        f.write(f"Epochs: {model_info['Epochs']}, Batch Size: {model_info['Batch Size']}, Learning Rate: {model_info['Learning Rate']}\n")
        f.write(f"Training Time: {model_info['Training Time (seconds)']} seconds\n")
        f.write(f"Final Loss: {model_info['Final Loss']}\n")
        f.write(f"Classes: {', '.join(model_info['Classes'])}\n")
        f.write(f"Dataset Size: {model_info['Dataset Size']} images\n")
        f.write(f"Model Path: {model_info['Model Path']}\n")
        f.write("=" * 50 + "\n")


if __name__ == "__main__":
    data_path = "images/dataset(4.4k)"  # Path to training dataset
    model_name = "model12"
    train_model(
        data_path=data_path,
        model_name=model_name,
        n_modes_height=16,
        n_modes_width=16,
        hidden_channels=64,
        batch_size=8,
        lr=0.0005,
        epochs=5,
        patience=3
    )

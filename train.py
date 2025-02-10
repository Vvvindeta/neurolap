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


def train_model(data_path, model_name, n_modes_height, n_modes_width, hidden_channels,
                batch_size, lr, epochs, load_model_path=None):
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

    start_epoch = 0
    if load_model_path:
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from '{load_model_path}', resuming from epoch {start_epoch}")

    start_time = time.time()

    os.makedirs("train_log", exist_ok=True)
    log_file_path = os.path.join("train_log", f"train_log_{model_name}.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Training {'resumed' if load_model_path else 'started'} for model: {model_name}\n")
        if load_model_path:
            log_file.write(f"Parent model: {load_model_path}\n")
        log_file.write(
            f"Parameters: {{'n_modes_height': {n_modes_height}, 'n_modes_width': {n_modes_width}, 'hidden_channels': {hidden_channels}}}\n")
        log_file.write(f"Total Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}\n")
        log_file.write("=" * 50 + "\n")

    for epoch in range(start_epoch, epochs):
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

            with open(log_file_path, "a") as log_file:
                log_file.write(
                    f"Epoch {epoch + 1}/{epochs}, Batch {batch_index}/{len(dataloader)}, Loss: {loss.item():.4f}\n")

        avg_loss = epoch_loss / len(dataloader)
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Epoch {epoch + 1}/{epochs} completed, Average Loss: {avg_loss:.4f}\n")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Сохраняем модель после каждой эпохи
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'current_loss': avg_loss
        }
        torch.save(checkpoint, f"made_model/{model_name}.pth")

    training_time = time.time() - start_time

    with open(log_file_path, "a") as log_file:
        log_file.write("Training completed.\n")
        log_file.write("=" * 50 + "\n")

    model_info = {
        "Model Name": model_name,
        "Parent Model": load_model_path if load_model_path else "None",
        "Parameters": {
            "n_modes_height": n_modes_height,
            "n_modes_width": n_modes_width,
            "hidden_channels": hidden_channels,
        },
        "Total Epochs": epochs,
        "Trained Epochs": epochs - start_epoch,
        "Batch Size": batch_size,
        "Learning Rate": lr,
        "Training Time (seconds)": round(training_time, 2),
        "Classes": dataset.classes,
        "Dataset Size": len(dataset),
        "Model Path": f"made_model/{model_name}.pth"
    }

    os.makedirs("made_model", exist_ok=True)
    models_file = os.path.join("made_model", "models2.txt")

    with open(models_file, "a") as f:
        f.write(f"Model Name: {model_info['Model Name']}\n")
        if load_model_path:
            f.write(f"Parent Model: {model_info['Parent Model']}\n")
        f.write(f"Parameters: {model_info['Parameters']}\n")
        f.write(f"Total Epochs: {model_info['Total Epochs']}, Trained Epochs: {model_info['Trained Epochs']}\n")
        f.write(f"Batch Size: {model_info['Batch Size']}, Learning Rate: {model_info['Learning Rate']}\n")
        f.write(f"Training Time: {model_info['Training Time (seconds)']} seconds\n")
        f.write(f"Classes: {', '.join(model_info['Classes'])}\n")
        f.write(f"Dataset Size: {model_info['Dataset Size']} images\n")
        f.write(f"Model Path: {model_info['Model Path']}\n")
        f.write("=" * 50 + "\n")


if __name__ == "__main__":
    train_model(
        data_path="images/dataset(4.4k)",
        model_name="model34",
        n_modes_height=16,
        n_modes_width=16,
        hidden_channels=128,
        batch_size=8,
        lr=0.0015,
        epochs=3,
        load_model_path=None
    )

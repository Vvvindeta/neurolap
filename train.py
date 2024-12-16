import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from neuralop.models.fno import FNO2d

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


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
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


def train_model(dataset_path, model_save_path, num_samples_per_class=None, epochs=10, batch_size=8):
    print("Loading dataset...")
    train_dataset = datasets.ImageFolder(dataset_path, transform=transform)

    if num_samples_per_class is not None:
        print(f"Filtering dataset to limit {num_samples_per_class} samples per class...")
        class_counts = {cls: 0 for cls in range(len(train_dataset.classes))}
        filtered_indices = []
        for idx, (_, label) in enumerate(train_dataset):
            if class_counts[label] < num_samples_per_class:
                filtered_indices.append(idx)
                class_counts[label] += 1
        train_dataset = torch.utils.data.Subset(train_dataset, filtered_indices)

    print("Creating data loader...")
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(train_dataset.dataset.classes) if isinstance(train_dataset, torch.utils.data.Subset) else len(
        train_dataset.classes)
    print(f"Initializing model with {num_classes} classes...")
    model = FNOImageClassifier(n_modes_height=16, n_modes_width=16, hidden_channels=100, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

    print("Saving model...")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


# Пример использования
if __name__ == "__main__":
    dataset_path = "images/dataset"  # Укажите путь к данным
    model_save_path = "fno_image_classifier.pth"
    train_model(dataset_path, model_save_path, num_samples_per_class=500, epochs=5, batch_size=8)

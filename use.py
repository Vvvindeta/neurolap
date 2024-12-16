import torch
from torchvision import datasets
from torchvision.transforms import transforms
import os
from collections import defaultdict
from train import FNOImageClassifier


def load_model(model_path, num_classes):
    model = FNOImageClassifier(n_modes_height=16, n_modes_width=16, hidden_channels=100, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_model(model, dataset_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    classification_count = defaultdict(lambda: defaultdict(int))

    print("Starting evaluation...")
    for inputs, labels in dataloader:
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

        label_name = dataset.classes[labels.item()]
        predicted_name = dataset.classes[predicted.item()]

        print(f"True label: {label_name}, Predicted label: {predicted_name}")

        class_total[label_name] += 1
        classification_count[label_name][predicted_name] += 1
        if predicted == labels:
            correct += 1
            class_correct[label_name] += 1
        total += 1

    print("\nEvaluation Summary:")
    for class_name in dataset.classes:
        class_count = class_total[class_name]
        correct_count = class_correct[class_name]
        class_accuracy = (correct_count / class_count * 100) if class_count > 0 else 0
        print(f"Class: {class_name}, Total: {class_count}, Correct: {correct_count}, Accuracy: {class_accuracy:.2f}%")

        print("Classifications:")
        for predicted_name, count in classification_count[class_name].items():
            print(f"  Predicted as {predicted_name}: {count} times")

    overall_accuracy = correct / total * 100
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    print(f"Total images processed: {total}")
    print(f"Total correctly classified images: {correct}")
    print(f"Number of unique classes evaluated: {len(dataset.classes)}")


if __name__ == "__main__":
    dataset_path = "training_dataset(12)"  # Specify the dataset path
    model_path = "made_model/model2.pth"  # Specify the path to the saved model

    class_names = datasets.ImageFolder(dataset_path).classes
    model = load_model(model_path, num_classes=len(class_names))

    evaluate_model(model, dataset_path)

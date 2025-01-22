import os
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from collections import defaultdict
from train import FNOImageClassifier


def load_available_models(models_file_path):
    models = []
    if not os.path.exists(models_file_path):
        print("No models available. Ensure the models.txt file exists and has entries.")
        return models

    with open(models_file_path, "r") as file:
        lines = file.readlines()

    current_model = {}
    for line in lines:
        if line.startswith("Model Name:"):
            if current_model:
                models.append(current_model)
                current_model = {}
            current_model["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Parameters:"):
            current_model["params"] = eval(line.split(":", 1)[1].strip())
        elif line.startswith("Model Path:"):
            current_model["path"] = line.split(":", 1)[1].strip()
        elif line.startswith("=" * 50):
            if current_model:
                models.append(current_model)
                current_model = {}
    return models


def select_model(models):
    print("Available Models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']} - {model['params']}")
    while True:
        try:
            choice = int(input("Select a model by number: ")) - 1
            if 0 <= choice < len(models):
                return models[choice]
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Enter a valid number.")


def load_model(model_path, num_classes, model_params):
    model = FNOImageClassifier(
        n_modes_height=model_params["n_modes_height"],
        n_modes_width=model_params["n_modes_width"],
        hidden_channels=model_params["hidden_channels"],
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def evaluate_model(model, dataset_path, log_file_name):
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
    classification_counts = defaultdict(lambda: defaultdict(int))

    os.makedirs("use_log", exist_ok=True)
    with open(log_file_name, "a") as log_file:
        log_file.write(f"Testing dataset: {dataset_path}\n")
        log_file.write(f"Classes: {dataset.classes}\n")
        log_file.write("=" * 50 + "\n")

        print("Starting evaluation...")
        for inputs, labels in dataloader:
            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

            label_name = dataset.classes[labels.item()]
            predicted_name = dataset.classes[predicted.item()]

            log_file.write(f"True label: {label_name}, Predicted label: {predicted_name}\n")
            classification_counts[label_name][predicted_name] += 1

            class_total[label_name] += 1
            if predicted == labels:
                correct += 1
                class_correct[label_name] += 1
            total += 1

        log_file.write("\nEvaluation Summary:\n")
        for class_name in dataset.classes:
            class_count = class_total[class_name]
            correct_count = class_correct[class_name]
            class_accuracy = (correct_count / class_count * 100) if class_count > 0 else 0
            log_file.write(
                f"Class: {class_name}, Total: {class_count}, Correct: {correct_count}, Accuracy: {class_accuracy:.2f}%\n")

        log_file.write("\nClassification Details:\n")
        for true_class, predictions in classification_counts.items():
            log_file.write(f"True Class: {true_class}\n")
            for predicted_class, count in predictions.items():
                log_file.write(f"  Predicted as {predicted_class}: {count} times\n")

        overall_accuracy = correct / total * 100
        log_file.write(f"\nOverall Accuracy: {overall_accuracy:.2f}%\n")
        log_file.write(f"Total images processed: {total}\n")
        log_file.write(f"Total correctly classified images: {correct}\n")
        log_file.write(f"Number of unique classes evaluated: {len(dataset.classes)}\n")
        log_file.write("=" * 50 + "\n")

    print("Evaluation complete. Check log file for details.")


if __name__ == "__main__":
    models_file_path = "made_model/models.txt"
    models = load_available_models(models_file_path)
    if not models:
        exit()

    selected_model = select_model(models)
    dataset_root = "images"

    print("\nAvailable Test Datasets:")
    dataset_folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    for i, folder in enumerate(dataset_folders, 1):
        print(f"{i}. {folder}")

    while True:
        try:
            dataset_choice = int(input("Select a dataset by number: ")) - 1
            if 0 <= dataset_choice < len(dataset_folders):
                selected_dataset = dataset_folders[dataset_choice]
                break
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Enter a valid number.")

    dataset_path = os.path.join(dataset_root, selected_dataset)
    log_file_name = os.path.join("use_log", f"use_log_{selected_model['name']}_{selected_dataset}.txt")

    model = load_model(selected_model["path"], num_classes=len(os.listdir(dataset_path)),
                       model_params=selected_model["params"])
    evaluate_model(model, dataset_path, log_file_name)
    
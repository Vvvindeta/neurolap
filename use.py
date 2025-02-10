import os
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from collections import defaultdict
from train2 import FNOImageClassifier


def load_available_models(models_file_path):
    models = []
    if not os.path.exists(models_file_path):
        print("No models available. Ensure the models.txt file exists and has entries.")
        return models

    with open(models_file_path, "r") as file:
        lines = file.readlines()

    current_model = {}
    for line in lines:
        line = line.strip()
        if line.startswith("Model Name:"):
            if current_model:
                models.append(current_model)
                current_model = {}
            current_model["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Parameters:"):
            current_model["params"] = eval(line.split(":", 1)[1].strip())
        elif line.startswith("Classes:"):
            classes = line.split(":", 1)[1].strip().split(", ")
            current_model["classes"] = classes
            current_model["num_classes"] = len(classes)
        elif line.startswith("Model Path:"):
            current_model["path"] = line.split(":", 1)[1].strip()
        elif line.startswith("=" * 50):
            if current_model:
                models.append(current_model)
                current_model = {}

    if current_model:
        models.append(current_model)

    return models


def select_model(models):
    print("\nAvailable Models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")
        print(f"   Classes: {', '.join(model['classes'])}")
        print(f"   Parameters: {model['params']}")
        print()
    while True:
        try:
            choice = int(input("Select a model by number: ")) - 1
            if 0 <= choice < len(models):
                return models[choice]
            print("Invalid choice. Try again.")
        except ValueError:
            print("Enter a valid number.")


def load_model(model_path, model_params, num_classes):
    # Инициализация модели с параметрами из чекпоинта
    model = FNOImageClassifier(
        n_modes_height=model_params['n_modes_height'],
        n_modes_width=model_params['n_modes_width'],
        hidden_channels=model_params['hidden_channels'],
        num_classes=num_classes
    )

    # Загрузка полного чекпоинта
    checkpoint = torch.load(model_path)

    # Загрузка состояния модели
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_model(model, dataset_path, class_names, log_file_name):
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
        log_file.write(f"Model Classes: {', '.join(class_names)}\n")
        log_file.write(f"Dataset Classes: {dataset.classes}\n")
        log_file.write("=" * 50 + "\n")

        print("Starting evaluation...")
        for inputs, labels in dataloader:
            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

            true_class = dataset.classes[labels.item()]
            predicted_class = dataset.classes[predicted.item()]

            log_file.write(f"True: {true_class}, Predicted: {predicted_class}\n")
            classification_counts[true_class][predicted_class] += 1

            class_total[true_class] += 1
            if predicted == labels:
                correct += 1
                class_correct[true_class] += 1
            total += 1

        log_file.write("\nEvaluation Summary:\n")
        for class_name in dataset.classes:
            class_count = class_total.get(class_name, 0)
            correct_count = class_correct.get(class_name, 0)
            class_accuracy = (correct_count / class_count * 100) if class_count > 0 else 0
            log_file.write(
                f"Class: {class_name}, Total: {class_count}, Correct: {correct_count}, Accuracy: {class_accuracy:.2f}%\n")

        log_file.write("\nClassification Matrix:\n")
        for true_class in dataset.classes:
            log_file.write(f"True Class: {true_class}\n")
            for pred_class in dataset.classes:
                count = classification_counts[true_class].get(pred_class, 0)
                log_file.write(f"  {pred_class}: {count}\n")

        overall_accuracy = correct / total * 100 if total > 0 else 0
        log_file.write(f"\nOverall Accuracy: {overall_accuracy:.2f}%\n")
        log_file.write(f"Total Images: {total}\n")
        log_file.write(f"Correct Predictions: {correct}\n")
        log_file.write("=" * 50 + "\n")

    print(f"Evaluation complete. Results saved to {log_file_name}")


if __name__ == "__main__":
    models_file_path = "made_model/models2.txt"
    models = load_available_models(models_file_path)
    if not models:
        exit()

    selected_model = select_model(models)

    print("\nAvailable Test Datasets:")
    dataset_root = "images"
    dataset_folders = [f for f in os.listdir(dataset_root)
                       if os.path.isdir(os.path.join(dataset_root, f))]

    for i, folder in enumerate(dataset_folders, 1):
        print(f"{i}. {folder}")

    dataset_choice = -1
    while dataset_choice < 0 or dataset_choice >= len(dataset_folders):
        try:
            dataset_choice = int(input("Select dataset by number: ")) - 1
        except ValueError:
            continue

    dataset_path = os.path.join(dataset_root, dataset_folders[dataset_choice])
    log_file_name = os.path.join("use_log",
                                 f"eval_{selected_model['name']}_{dataset_folders[dataset_choice]}.txt")

    # Загрузка модели с правильным количеством классов
    model = load_model(
        model_path=selected_model['path'],
        model_params=selected_model['params'],
        num_classes=selected_model['num_classes']
    )

    # Проверка совместимости классов
    test_dataset = datasets.ImageFolder(dataset_path)
    if set(test_dataset.classes) != set(selected_model['classes']):
        print(
            f"Warning: Model classes ({selected_model['classes']}) don't match test dataset classes ({test_dataset.classes})")

    evaluate_model(
        model=model,
        dataset_path=dataset_path,
        class_names=selected_model['classes'],
        log_file_name=log_file_name
    )

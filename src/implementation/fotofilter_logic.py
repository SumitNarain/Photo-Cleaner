"""
Dataset: Blur dataset
    https://www.kaggle.com/datasets/kwentar/blur-dataset
    Contains 1050 images that contain sharp images, blurred images, and motion-blurred images
    Current implementation will remove the pictures but future implementation can
    use AI to improve the blurred photos and then run though the duplicate photos model as well
    to choose the best photo(s) and remove the rest into a trash folder
"""

import kagglehub
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

# NOTE: These imports need the SimpleCNN and FotoBlurryDataset classes to be defined
# or importable from these locations.
from model.convolution_neural_network import SimpleCNN
from src.data.FotoBlurryDataset import FotoBlurryDataset, data_transforms

# --- Global Constants ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 5  # For Early Stopping

# --- Execution Entry Point ---
# This block ensures that the code inside it only runs when the script is executed directly,
# preventing child processes from re-running the setup code.
if __name__ == '__main__':

    print("Starting script execution...")

    # --- 1. Data Setup ---
    # Download the latest version
    path = kagglehub.dataset_download("kwentar/blur-dataset")
    print("Path to dataset files:", path)

    # Ensure Device agnostic code:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    foto_dataset = FotoBlurryDataset(data_path=path, transform=data_transforms)

    # Note: Removed the unused 'dataloader = DataLoader(...)' line.

    training_size = int(0.8 * len(foto_dataset))
    validation_size = len(foto_dataset) - training_size
    train_dataset, validation_dataset = random_split(
        foto_dataset, [training_size, validation_size]
    )

    # DataLoaders (where num_workers > 0 causes the issue)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 2. Model and Optimizer Setup ---
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    # Using the variable LEARNING_RATE for consistency
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Early Stopping Variables ---
    min_val_loss = float('inf')
    patience_counter = 0

    print("Starting training...")
    print("-" * 30)

    # --- 4. Training Loop with Early Stopping ---
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct_predictions / total_samples

        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # --- Early Stopping Check ---
        if avg_val_loss < min_val_loss:
            # Save model if validation loss improves
            min_val_loss = avg_val_loss
            patience_counter = 0

            # Saving the state_dict for the best model
            torch.save(model.state_dict(), 'best_blur_classifier.pth')
            print(f"    --> Saved best model (Val Loss: {min_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"    --> Val Loss did not improve. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("\n*** Early stopping triggered! Training complete. ***")
                break  # Exit the loop

    print("-" * 30)
    print("Loading best model weights for final use...")
    # Load the best saved model state before any final testing
    model.load_state_dict(torch.load('best_blur_classifier.pth'))
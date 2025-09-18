from ultralytics import YOLO
import torch

def train_basketball_model():
    """
    This function trains a YOLOv12n model on a custom basketball dataset.
    """
    # Path to dataset's yaml file - a dictionary for classes and train/split/test
    dataset_yaml_path = 'yolo_dataset/data.yaml'
    # Base weights that will now be tuned specifically for our Basketball dataset
    model_name = "yolov12n.pt"
    # Training parameters
    # Number if times to go through complete dataset
    epochs = 200
    # Resize all images to this size before training
    image_size = 640
    # Number of images to process at once.
    batch_size = 64
    project_name = 'Basketball_Detection'
    run_name = f'{model_name}_{epochs}_epochs_{batch_size}_batch_size'
    # Load YOLOv12n model
    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    print(f"Starting training for {epochs} epochs...")
    try:
        results = model.train(
            data=dataset_yaml_path,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project=project_name,
            name=run_name,
            device=device,
            # Stop training early if no improvement is seen after 10 epochs
            patience=20
        )
        print("--- Training complete! ---")
        print(f"Results, weights, and plots saved to: '{project_name}/{run_name}'")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Please ensure your 'data.yaml' file is configured correctly and the dataset paths are valid.")
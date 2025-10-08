import os
import zipfile
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from BasketballAnalyser import *
from YOLO_model import *
import torch
from ultralytics import YOLO
from DETR_data_converter_utils import *
from DETR_model import *
from SVM_HOG_model import *
BBALL_DATASET = "yolo_dataset"
from MOT_SIFT import *

def unzip_dataset(zip_path, extract_path):
    """Unzips the dataset file."""
    print(f"Unzipping '{zip_path}' to '{extract_path}'...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Unzipping complete.")
        return True
    except FileNotFoundError:
        print(f"Error: The file '{zip_path}' was not found.")
        return False
    except zipfile.BadZipFile:
        print(f"Error: The file '{zip_path}' is not a valid zip file.")
        return False


def analyse_and_plot_split(split_name, label_dir, class_names):
    """
    Analyses class distribution for a single data split and generates a plot.
    """
    print(f"\nAnalysing Class Distribution for '{split_name}' set")

    if not os.path.exists(label_dir):
        print(f"Warning: Label directory not found at '{label_dir}'. Skipping.")
        return

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    if not label_files:
        print(f"No label files found in '{label_dir}'. Skipping.")
        return

    class_counts = {name: 0 for name in class_names}

    for filename in label_files:
        file_path = os.path.join(label_dir, filename)
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    idx = int(line.split()[0])
                    if 0 <= idx < len(class_names):
                        class_counts[class_names[idx]] += 1
                except (ValueError, IndexError):
                    print(f"Warning: Skipping malformed line in {file_path}")

    # Display and plot the results for this split
    df = pd.DataFrame(list(class_counts.items()), columns=['Class Name', 'Instance Count'])
    print(df.to_string(index=False))

    df_sorted = df.sort_values('Instance Count', ascending=True)
    plt.figure(figsize=(10, len(class_names) * 0.4 + 2))
    plt.barh(df_sorted['Class Name'], df_sorted['Instance Count'], color='teal')
    plt.xlabel('Instance Count')
    plt.ylabel('Class Name')
    plt.title(f'YOLO Class Distribution for {split_name.capitalize()} Set')
    plt.tight_layout()

    plot_filename = f'distribution_{split_name}.png'
    plt.savefig(plot_filename)
    print(f"Analysis chart for '{split_name}' saved as '{plot_filename}'.")
    plt.show()


def analyse_yolo_dataset(dataset_dir):
    """
    Analyses the unzipped YOLO dataset, splitting the analysis by train/val/test.
    """
    yaml_path = None
    for root, dirs, files in os.walk(dataset_dir):
        if 'data.yaml' in files:
            yaml_path = os.path.join(root, 'data.yaml')
            break

    if yaml_path is None:
        print("\nError: Could not find 'data.yaml'.")
        return

    dataset_root = os.path.dirname(yaml_path)

    with open(yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)

    class_names = data_yaml.get('names', [])
    if not class_names:
        print("Error: No class names found in data.yaml.")
        return

    print(f"\nDataset Info \nClass names: {class_names}")

    print("\nSplit Analysis")
    # Loop through each split defined in the YAML file
    for split_name in ['train', 'val', 'test']:
        if split_name in data_yaml:
            # Build paths dynamically for each split
            image_dir = os.path.join(dataset_root, split_name, 'images')
            label_dir = os.path.join(dataset_root, split_name, 'labels')
            # Print image/label counts
            if os.path.exists(image_dir):
                n_img = len(os.listdir(image_dir))
                n_lbl = len([f for f in os.listdir(label_dir) if f.endswith('.txt')]) if os.path.exists(
                    label_dir) else 0
                print(f"\nFound {n_img} images and {n_lbl} labels in '{split_name}' set.")
                # Call the helper function to perform the detailed analysis and plotting
                analyse_and_plot_split(split_name, label_dir, class_names)
            else:
                print(f"\nWarning: No image directory found for '{split_name}' at '{image_dir}'")
        else:
            print(f"\nInfo: Split '{split_name}' not defined in data.yaml.")


def find_best_weights():
    """
    Evaluates models from a list of directories to find the best performing one.
    """
    # Path to the parent directory containing all trained models
    parent_dir = 'Basketball_Detection'
    # Define the list of model directories to evaluate
    model_dirs = [
        'yolov12n.pt_200_epochs_16_batch_size',
        'yolov12n.pt_200_epochs_64_batch_size',
        'yolov12n.pt_200_epochs_64_batch_size_augmented',
        'yolov12n.pt_200_epochs_64_batch_size_augmented_c&p',
        'yolov12n.pt_200_epochs_64_batch_size_augmented_c&p_mosaic',
        'yolov12n.pt_200_epochs_64_batch_size_augmented_c&p_mosaic_cls'
    ]

    best_mAP = -1.0
    best_model_path = None

    # Path to your dataset's validation split
    dataset_yaml_path = 'yolo_dataset/data.yaml'

    for model_dir in model_dirs:
        model_path = os.path.join(parent_dir, model_dir, 'weights', 'best.pt')

        if os.path.exists(model_path):
            print(f"\n--- Validating model: {model_path} ---")
            try:
                model = YOLO(model_path)
                metrics = model.val(data=dataset_yaml_path)
                current_mAP = metrics.results_dict['metrics/mAP50(B)']

                print(f"Model {model_dir} achieved mAP50(B) of {current_mAP:.4f}")

                if current_mAP > best_mAP:
                    best_mAP = current_mAP
                    best_model_path = model_path
            except Exception as e:
                print(f"An error occurred during validation of {model_path}: {e}")
        else:
            print(f"Model not found at: {model_path}")

    print(f"\n--- All models evaluated. ---")
    print(f"Best model found: {best_model_path} with an mAP50(B) of {best_mAP:.4f}")

if __name__ == "__main__":
    """
        Training YOLO player detector.
    """
    # train_basketball_model()
    """
        Out of all the YOLO training runs, the weights that produce the best metrics are kept for the live
        analyser.
    """
    # find_best_weights()
    """
        Converting YOLO dataset format into COCO for DETR.
    """
    # # Check device status (Standard practice for ML scripts)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Running on device: {device}")
    # # Define the path to the YOLO dataset configuration
    # DATASET_YAML_PATH = 'yolo_dataset/data.yaml'
    # # Load categories from the YAML file
    # BASKETBALL_CATEGORIES = load_categories_from_yaml(DATASET_YAML_PATH)
    # if not BASKETBALL_CATEGORIES:
    #     print("\nFATAL ERROR: Could not load categories from YAML. Conversion halted.")
    # else:
    #     print(f"Successfully loaded {len(BASKETBALL_CATEGORIES)} Categories: {BASKETBALL_CATEGORIES}")
    #     # Define the splits to process
    #     SPLITS = ['train', 'val', 'test']
    #     print("\nStarting Conversion for All Splits")
    #     # Create output directory for JSONs if it doesn't exist
    #     os.makedirs('./detr_annotations', exist_ok=True)
    #     # Run conversion for all splits
    #     for split_name in SPLITS:
    #         # The run_split_conversion function handles paths and calls the converter.
    #         # It uses the default base_dir='./yolo_dataset' and outputs JSONs
    #         # to the main directory (e.g., ./detr_train_annotations.json).
    #         run_split_conversion(split_name, BASKETBALL_CATEGORIES)
    #
    #     print("\nAll dataset splits have been successfully converted to COCO JSON format.")

    """
        Train the DETR
    """
    # train_detr_model(epochs=200, batch_size=8)
    """
        Train the SVM-HOG
    """
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # dataset_base_path = os.path.join(script_dir, 'yolo_dataset')
    #
    # # Training paths
    # train_image_dirs = [os.path.join(dataset_base_path, 'train', 'images'),
    #                     os.path.join(dataset_base_path, 'val', 'images')]
    # train_label_dirs = [os.path.join(dataset_base_path, 'train', 'labels'),
    #                     os.path.join(dataset_base_path, 'val', 'labels')]
    #
    # # Validation paths
    # val_image_dirs = [os.path.join(dataset_base_path, 'val', 'images')]
    # val_label_dirs = [os.path.join(dataset_base_path, 'val', 'labels')]
    #
    # # Test paths
    # test_image_dirs = [os.path.join(dataset_base_path, 'test', 'images')]
    # test_label_dirs = [os.path.join(dataset_base_path, 'test', 'labels')]
    #
    # # Initialise and train the SVM-HOG
    # print("\nTraining the SVM-HOG Detector")
    # # A rectangular shape to best identify the players
    # detector = SVM_HOG_ObjectDetector(window_size=(64, 128))
    # detector.train(image_dirs=train_image_dirs, label_dirs=train_label_dirs)
    #
    # # Evaluate SVM-HOG performance
    # print("Evaluating SVM-HOG")
    #
    # print("\nTraining Scores")
    # detector.evaluate_model(train_image_dirs, train_label_dirs)
    #
    # print("\nValidation Scores")
    # detector.evaluate_model(val_image_dirs, val_label_dirs)
    #
    # print("\nTest Scores")
    # detector.evaluate_model(test_image_dirs, test_label_dirs)
    #
    # # Run detection on a single test image
    # print("Running Full Object Detection")
    #
    # test_image_dir = test_image_dirs[0]  # Use the first (and only) test directory
    # test_label_dir = test_label_dirs[0]
    # all_test_images = os.listdir(test_image_dir)
    #
    # if not all_test_images:
    #     print("No images found in the test directory.")
    # else:
    #     # Pick a random image from the test set
    #     test_image_name = random.choice(all_test_images)
    #     test_image_path = os.path.join(test_image_dir, test_image_name)
    #     print(f"Randomly selected test image: {test_image_path}")
    #
    #     # Run detection and visualise predicted boxes (in green)
    #     predicted_boxes = detector.detect(test_image_path, confidence_threshold=0.5)
    #     print(f"\nDetected {len(predicted_boxes)} objects (predictions).")
    #
    #     # Load the image to draw predictions on
    #     predicted_image = cv2.imread(test_image_path)
    #     for box in predicted_boxes:
    #         x1, y1, x2, y2 = box[:4]
    #         # Draw a green rectangles for predictions
    #         cv2.rectangle(predicted_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    #     # Save the prediction visualisation
    #     prediction_output_path = f'prediction_{test_image_name}'
    #     cv2.imwrite(prediction_output_path, predicted_image)
    #     print(f"Prediction visualization saved to '{prediction_output_path}'")
    #
    #     # Load the true labels and visualise ground truth boxes (in blue)
    #     # Load a fresh copy of the image for ground truth
    #     gt_image = cv2.imread(test_image_path)
    #     h, w, _ = gt_image.shape
    #
    #     # Find the corresponding label file
    #     basename, _ = os.path.splitext(test_image_name)
    #     label_path = os.path.join(test_label_dir, basename + '.txt')
    #
    #     if not os.path.exists(label_path):
    #         print(f"No ground truth label file found for this image at '{label_path}'")
    #     else:
    #         with open(label_path, 'r') as f:
    #             lines = f.readlines()
    #             print(f"Found {len(lines)} objects in ground truth file.")
    #             for line in lines:
    #                 # Parse the YOLO format line
    #                 class_id, x_center, y_center, width, height = map(float, line.split())
    #
    #                 # De-normalise coordinates back to pixel values
    #                 x_center_px = x_center * w
    #                 y_center_px = y_center * h
    #                 width_px = width * w
    #                 height_px = height * h
    #
    #                 # Calculate top-left and bottom-right corners
    #                 x1 = int(x_center_px - (width_px / 2))
    #                 y1 = int(y_center_px - (height_px / 2))
    #                 x2 = int(x_center_px + (width_px / 2))
    #                 y2 = int(y_center_px + (height_px / 2))
    #
    #                 # Draw a blue rectangle for ground truth
    #                 cv2.rectangle(gt_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #
    #         # Save the ground truth visualisation
    #         gt_output_path = f'ground_truth_{test_image_name}'
    #         cv2.imwrite(gt_output_path, gt_image)
    #         print(f"Ground truth visualization saved to '{gt_output_path}'")

    """
       Run YOLOv12n Analyser with ByteTrack or BoT-SORT
    """

    # MODEL_PATH = "Basketball_Detection/yolov12n.pt_200_epochs_64_batch_size_augmented/weights/best.pt"
    # VIDEO_SOURCE = "https://www.youtube.com/watch?v=6OTIqjh0eKc&list=PLiVlTTnDnAcsX_H-K9sy98OKvjGtpYSi-&index=1&t=443s"
    # START_TIME = "7:00"
    # try:
    #     analyser = BasketballAnalyser(
    #         model_path=MODEL_PATH,
    #         video_source=VIDEO_SOURCE,
    #         start_time=START_TIME
    #     )
    #     analyser.process_video()
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    """
       Run Basketball Analysis with the custom MOT_SIFT Tracker
    """

    MODEL_PATH = "Basketball_Detection/yolov12n.pt_200_epochs_64_batch_size_augmented/weights/best.pt"
    VIDEO_SOURCE = "https://www.youtube.com/watch?v=6OTIqjh0eKc&list=PLiVlTTnDnAcsX_H-K9sy98OKvjGtpYSi-&index=1&t=443s"
    START_TIME = "7:00"

    try:
        # 1. Create an instance of your custom SIFT tracker
        sift_tracker_config = MOT_SIFT(
            detection_conf=0.4,
            sift_good_dist=300.0,
            min_sift_score=20.0,
            accumulate_sift=3,
            max_age=30
        )

        # 2. Create an instance of the BasketballAnalyser
        #    and "inject" the sift_tracker into it.
        analyser = BasketballAnalyser(
            model_path=MODEL_PATH,  # Still needed for the fallback
            video_source=VIDEO_SOURCE,
            start_time=START_TIME,
            tracker_config=sift_tracker_config  # <-- FEED THE TRACKER HERE
        )

        # 3. Run the analysis
        analyser.process_video()

    except Exception as e:
        print(f"An error occurred: {e}")
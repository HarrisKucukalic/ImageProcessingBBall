import os
import zipfile
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from PlayerStats import *
from StatsRecorder import *
from BasketballAnalyser import *
from YOLO_model import *
import torch

BBALL_DATASET = "yolo_dataset"

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
    print(f"📊 Analysis chart for '{split_name}' saved as '{plot_filename}'.")
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

    print(f"\n--- Dataset Info ---\nClass names: {class_names}")

    print("\n--- Split Analysis ---")
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


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    # train_basketball_model()
    MODEL_PATH = "Basketball_Detection/yolov12n.pt_200_epochs_16_batch_size/weights/best.pt"
    VIDEO_SOURCE = "https://www.youtube.com/watch?v=d_JI-QGcpgI"  # Example YouTube URL

    # Default values that were in the parser
    TRACKER_CONFIG = 'bytetrack.yaml'
    CONF_THRESH = 0.4
    IOU_THRESH = 0.7
    try:
        analyser = BasketballAnalyser(
            model_path=MODEL_PATH,
            video_source=VIDEO_SOURCE,
            tracker_config=TRACKER_CONFIG,
            conf_thresh=CONF_THRESH,
            iou_thresh=IOU_THRESH
        )
        analyser.process_video()
    except Exception as e:
        print(f"An error occurred: {e}")
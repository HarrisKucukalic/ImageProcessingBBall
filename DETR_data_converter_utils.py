import os
import json
import glob
from PIL import Image
from tqdm import tqdm
import ast

def load_categories_from_yaml(yaml_path: str) -> dict:
    """
    Parses a YOLO data.yaml file to extract the class name mapping
    from the 'names:' section. Returns categories or an empty dictionary on failure.

    This function uses simple file parsing suitable for the standard YOLO format,
    avoiding external dependencies like PyYAML.
    """
    categories = {}
    print(f"Attempting to load categories from: {yaml_path}")

    try:
        with open(yaml_path, 'r') as f:
            lines = f.readlines()

        in_names_section = False

        for line in lines:
            line = line.strip()

            if line.startswith('names:'):
                in_names_section = True
                # Handle single-line names: names: [Player, Ball]
                if '[' in line and ']' in line:
                    names_str = line[line.find('['):line.find(']') + 1].strip()
                    names_list = ast.literal_eval(names_str)
                    return {i: name for i, name in enumerate(names_list)}
                continue

            if in_names_section:
                # Handle multi-line dictionary/list format (e.g., 0: Player)
                if ':' in line and not line.startswith('train') and not line.startswith('val') and not line.startswith(
                        'nc'):
                    try:
                        key, value = line.split(':', 1)
                        key = key.strip().lstrip('-').strip()
                        value = value.strip().strip("'\"")

                        if key.isdigit():
                            categories[int(key)] = value

                    except ValueError:
                        break  # Stop parsing if the line format breaks
                elif not line or line.startswith(('train', 'val', 'nc')):
                    break  # End of the 'names' block

    except FileNotFoundError:
        print(f"Error: YAML file not found at {yaml_path}. Returning empty categories.")
        return {}
    except Exception as e:
        print(f"Error: Failed to parse YAML at {yaml_path}. Exception: {e}. Returning empty categories.")
        return {}

    return categories

def convert_yolo_to_coco_json(yolo_image_dir: str, yolo_label_dir: str, output_json_path: str, categories: dict):
    """
    Converts YOLO-formatted dataset annotations (normalised xc, yc, w, h)
    into a COCO-like JSON structure (absolute xmin, ymin, w, h) suitable for DETR training.

    Args:
        yolo_image_dir (str): Path to the directory containing images.
        yolo_label_dir (str): Path to the directory containing YOLO .txt labels.
        output_json_path (str): Path where the final COCO JSON file will be saved.
        categories (dict): Dictionary mapping {class_index (int): class_name (str)}.
    """

    # Initialise COCO structure components
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": k, "name": v} for k, v in categories.items()]
    }

    # Helper variables
    image_id = 0
    annotation_id = 0

    # Get all image files
    image_paths = sorted(glob.glob(os.path.join(yolo_image_dir, '*.jpg')) +
                         glob.glob(os.path.join(yolo_image_dir, '*.png')))

    if not image_paths:
        print(f"Error: No images found in {yolo_image_dir}. Exiting.")
        return

    print(f"Found {len(image_paths)} images. Starting conversion...")

    for img_path in tqdm(image_paths, desc="Converting annotations"):

        # Get image metadata
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Skipping corrupt image {img_path}: {e}")
            continue

        img_filename = os.path.basename(img_path)

        # Add image entry to COCO structure
        coco_output["images"].append({
            "id": image_id,
            "file_name": img_filename,
            "width": img_width,
            "height": img_height
        })

        # Find corresponding YOLO label file
        label_filename = img_filename.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(yolo_label_dir, label_filename)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    # Normalised YOLO box: [xc, yc, w, h]
                    xc, yc, w_norm, h_norm = parts[1:5]

                    # Convert normalised YOLO box to absolute COCO box [xmin, ymin, w_abs, h_abs]

                    # Absolute width and height
                    w_abs = w_norm * img_width
                    h_abs = h_norm * img_height

                    # Absolute x_min and y_min
                    x_min = (xc - w_norm / 2) * img_width
                    y_min = (yc - h_norm / 2) * img_height

                    # COCO format requires segmentation (empty list) and area
                    area = w_abs * h_abs

                    # Add annotation entry
                    coco_output["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        # Assumes class_id directly maps to the category index used in 'categories'
                        "bbox": [round(x_min), round(y_min), round(w_abs), round(h_abs)],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": []
                    })

                    annotation_id += 1

        image_id += 1

    # Save the resulting JSON file
    with open(output_json_path, 'w') as outfile:
        json.dump(coco_output, outfile, indent=4)

    print(f"\nConversion complete! COCO JSON saved to: {output_json_path}")
    print(f"Total annotations: {annotation_id}")


def run_split_conversion(split_name, categories, base_dir='./yolo_dataset'):
    """
    Sets up the paths and detr_runs the conversion function for a single dataset split.
    """
    YOLO_IMAGES_DIR = os.path.join(base_dir,  split_name, 'images')
    YOLO_LABELS_DIR = os.path.join(base_dir, split_name, 'labels')
    OUTPUT_JSON_FILE = f'./detr_annotations/detr_{split_name}_annotations.json'

    print(f"Running DETR Data Conversion for {split_name.upper()} Split.")

    convert_yolo_to_coco_json(
        yolo_image_dir=YOLO_IMAGES_DIR,
        yolo_label_dir=YOLO_LABELS_DIR,
        output_json_path=OUTPUT_JSON_FILE,
        categories=categories
    )
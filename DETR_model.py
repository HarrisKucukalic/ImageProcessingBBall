import os
import json
import torch
import time  # Import time for unique run naming
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
from tqdm import tqdm
# NOTE: To get the full P, R, F1, and Confusion Matrix results like YOLO,
# you should install a metric library like torchmetrics:
from torchmetrics.detection import MeanAveragePrecision

# from torchmetrics.classification import MulticlassConfusionMatrix

# --- Configuration ---
# Set the device for training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path to your image and annotation files
BASE_DATA_DIR = './yolo_dataset'
ANNOTATION_DIR = './detr_annotations'
TRAIN_JSON = os.path.join(ANNOTATION_DIR, 'detr_train_annotations.json')
VAL_JSON = os.path.join(ANNOTATION_DIR, 'detr_val_annotations.json')
TEST_JSON = os.path.join(ANNOTATION_DIR, 'detr_test_annotations.json')

# --- EXPERIMENT CONSISTENCY SETTING ---
# Set the image size to match the input resolution used for the YOLO model
# (e.g., 640, 960, 1280). This helps ensure fair comparisons.
IMAGE_SIZE = 640


# --- Helper Function: COCO Annotation Loading (ADJUSTED) ---

def read_coco_json(json_path, image_base_path):
    """
    Reads a COCO JSON annotation file and prepares a list of targets
    for the PyTorch Dataset, saving both raw COCO targets and processed tensors.
    """
    if not os.path.exists(json_path):
        print(f"Error: Annotation file not found at {json_path}")
        return []

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 1. Map image IDs to file paths and dimensions
    img_map = {img['id']: {
        'path': os.path.join(image_base_path, img['file_name']),
        'width': img['width'],
        'height': img['height']
    } for img in coco_data['images']}

    # 2. Group annotations by image ID
    annotations_by_img = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        # Store the RAW COCO format annotations: list of dicts
        annotations_by_img[img_id].append(ann)

        # 3. Create the final list of samples (image_path, raw_coco_target, processed_tensor_target)
    samples = []
    for img_id, img_info in img_map.items():
        anns = annotations_by_img.get(img_id, [])

        # --- A. Prepare Raw COCO Target for Processor (Input to Hugging Face) ---
        # The processor expects the original list of annotations for the image
        raw_coco_target = {
            'image_id': img_id,
            'annotations': anns
        }

        # --- B. Prepare Processed Tensor Target (Used for Loss/Metrics) ---

        boxes = []
        labels = []
        for ann in anns:
            # COCO bbox format: [xmin, ymin, width, height] (absolute pixels)
            xmin, ymin, w, h = ann['bbox']

            # Convert to normalized DETR format: [xc, yc, w, h] (normalized 0-1)
            xc = (xmin + w / 2) / img_info['width']
            yc = (ymin + h / 2) / img_info['height']
            w_norm = w / img_info['width']
            h_norm = h / img_info['height']

            boxes.append([xc, yc, w_norm, h_norm])
            labels.append(ann['category_id'])

            # Processed Tensor Target (Used by the training loop for metrics/loss verification)
        processed_target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'class_labels': torch.tensor(labels, dtype=torch.long),
            # DETR needs the original image size for post-processing
            'orig_size': torch.tensor([img_info['height'], img_info['width']])
        }

        samples.append({
            'image_path': img_info['path'],
            'raw_coco_target': raw_coco_target,  # <-- NEW: Used by processor
            'processed_target': processed_target  # <-- NEW: Used by loss/metrics
        })

    print(f"Loaded {len(samples)} samples from {json_path}")
    return samples


# --- PyTorch Dataset for DETR (ADJUSTED) ---

class DetrDataset(Dataset):
    """Custom PyTorch Dataset for DETR reading COCO JSON targets."""

    def __init__(self, coco_samples, processor):
        self.coco_samples = coco_samples
        self.processor = processor

    def __len__(self):
        return len(self.coco_samples)

    def __getitem__(self, idx):
        sample = self.coco_samples[idx]
        image = Image.open(sample['image_path']).convert("RGB")

        # raw_coco_target contains the image_id and list of annotations for the processor
        raw_coco_target = sample['raw_coco_target']

        # processed_target contains the PyTorch tensors for loss/metrics
        processed_target = sample['processed_target']

        # Apply processor, passing the RAW COCO format annotations.
        # The processor will do the resizing, normalization, and tensor conversion.
        encoding = self.processor(
            images=image,
            annotations=raw_coco_target,  # <-- PASSING RAW COCO TARGET
            return_tensors="pt"
        )

        # Return a dictionary containing the necessary tensors, which will be padded later
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'pixel_mask': encoding['pixel_mask'].squeeze(),
            'labels': processed_target,  # <-- USING the pre-calculated tensor targets
            'input_labels': encoding['labels'][0]  # Also return processor's labels if needed later
        }


# --- Custom Collate Function (RE-ADDED AND CORRECTED) ---
def detr_collate_fn_fixed(batch):
    """
    Custom collate function that handles padding of images and masks manually,
    resolving the RuntimeError: stack expects each tensor to be equal size.
    """
    # Separate the elements into lists
    pixel_values = [item['pixel_values'] for item in batch]
    pixel_mask = [item['pixel_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Determine max width and height in the batch
    max_h = max(p.shape[1] for p in pixel_values)
    max_w = max(p.shape[2] for p in pixel_values)

    padded_pixel_values = []
    padded_pixel_mask = []

    # Pad each image/mask to the max dimensions
    for img, mask in zip(pixel_values, pixel_mask):
        # Pad image tensor
        padding_h = max_h - img.shape[1]
        padding_w = max_w - img.shape[2]

        # Pad with 0s for the image (standard DETR padding)
        # Note: The padding order is (left, right, top, bottom) for a 3D tensor,
        # so we pad the last two dimensions (H and W).
        padded_img = torch.nn.functional.pad(img, (0, padding_w, 0, padding_h), value=0.0)
        padded_pixel_values.append(padded_img)

        # Pad mask tensor
        # Masks should be padded with 1s (indicating padding/no attention)
        padded_mask = torch.nn.functional.pad(mask, (0, padding_w, 0, padding_h), value=1.0)
        padded_pixel_mask.append(padded_mask)

    return {
        'pixel_values': torch.stack(padded_pixel_values),
        'pixel_mask': torch.stack(padded_pixel_mask),
        'labels': labels
    }


# --- Validation Function (FIXED: Removed num_classes arg) ---

def run_validation_epoch(model, dataloader, device, num_classes):
    """
    Runs one validation epoch to compute validation loss and mAP@50.
    """
    model.eval()
    total_val_loss = 0

    # Initialize metric calculator specifically for mAP@50
    # FIX: Removed 'num_classes=num_classes' due to compatibility error.
    val_metric_calculator = MeanAveragePrecision(
        box_format="cxcywh",
        iou_thresholds=[0.5],  # Only calculate mAP@50 for quick check
    ).to(device)

    # Disable gradient calculations during evaluation
    with torch.no_grad():
        for batch in dataloader:

            # The batch now contains nested lists/tensors due to the custom collate_fn.
            # We must explicitly extract the tensor data.

            inputs = {
                'pixel_values': batch['pixel_values'].to(device),
                'pixel_mask': batch['pixel_mask'].to(device),
                # Ensure labels are on the device for loss calculation
                'labels': [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                           for t in batch['labels']]
            }

            # --- 1. Get Model Outputs and Loss ---
            outputs = model(**inputs)
            loss = outputs.loss
            total_val_loss += loss.item()

            # --- 2. Format Predictions for mAP Calculation ---

            SCORE_THRESHOLD = 0.5
            preds_batch = []
            targets_batch = []

            for i in range(len(batch['labels'])):
                # Targets (using the labels passed to the model)
                targets_batch.append({
                    'boxes': batch['labels'][i]['boxes'].to(device),
                    'labels': batch['labels'][i]['class_labels'].to(device)
                })

                # Predictions
                logits = outputs.logits[i]
                scores = logits.softmax(-1)[:, :-1].max(-1).values
                labels = logits.argmax(-1)
                keep = scores >= SCORE_THRESHOLD

                preds_batch.append({
                    'boxes': outputs.pred_boxes[i][keep],
                    'scores': scores[keep],
                    'labels': labels[keep]
                })

            val_metric_calculator.update(preds_batch, targets_batch)

    # --- Compute Final Metrics ---
    avg_val_loss = total_val_loss / len(dataloader)
    metric_results = val_metric_calculator.compute()
    val_map_50 = metric_results["map_50"].item()

    # Reset metric state for next epoch
    val_metric_calculator.reset()

    return avg_val_loss, val_map_50


# --- Evaluation Function (Test Set) (FIXED: Removed num_classes arg) ---

def evaluate_model(model, dataloader, device, num_classes):
    """
    Runs model inference on a dataset (e.g., test split) and collects
    predictions for metric calculation, including mAP@50 and mAP@50:95.
    """
    print("\n--- Starting Model Evaluation ---")
    model.eval()

    # Initialize the MeanAveragePrecision metric for COCO metrics (mAP@50:95)
    # FIX: Removed 'num_classes=num_classes' due to compatibility error.
    metric_calculator = MeanAveragePrecision(
        box_format="cxcywh",
        iou_thresholds=None,  # Calculates 0.5:0.05:0.95
        class_metrics=True,
    ).to(device)

    # Disable gradient calculations during evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Test Set"):

            inputs = {
                'pixel_values': batch['pixel_values'].to(device),
                'pixel_mask': batch['pixel_mask'].to(device),
            }

            # --- 1. Get Model Outputs ---
            outputs = model(**inputs)

            # --- 2. Collect Targets and Predictions ---

            # Targets: Collect normalized boxes and labels, ensuring they are tensors on the device
            targets_batch = [{
                'boxes': t['boxes'].to(device),
                'labels': t['class_labels'].to(device)
            } for t in batch['labels']]

            SCORE_THRESHOLD = 0.5

            preds_batch = []
            for i in range(len(batch['labels'])):
                # Predictions
                logits = outputs.logits[i]
                scores = logits.softmax(-1)[:, :-1].max(-1).values
                labels = logits.argmax(-1)
                keep = scores >= SCORE_THRESHOLD

                preds_batch.append({
                    'boxes': outputs.pred_boxes[i][keep],
                    'scores': scores[keep],
                    'labels': labels[keep]
                })

            # --- 3. Update Metric Calculators ---
            metric_calculator.update(preds_batch, targets_batch)

    # --- 4. Compute and Extract Metrics ---
    print("\n--- Computing Metrics (mAP@50, mAP@50:95, P, R, F1) ---")

    metric_results = metric_calculator.compute()

    map_50_95 = metric_results["map"].item()
    map_50 = metric_results["map_50"].item()

    results = {
        'mAP@50:95 (COCO)': f"{map_50_95 * 100:.2f}%",
        'mAP@50': f"{map_50 * 100:.2f}%",
        'mAP_small': f"{metric_results['map_small'].item() * 100:.2f}%",
        'mAP_medium': f"{metric_results['map_medium'].item() * 100:.2f}%",
        'mAP_large': f"{metric_results['map_large'].item() * 100:.2f}%",
        'Recall (Max)': f"{metric_results['mar_100'].item() * 100:.2f}%",
        'Precision/Recall/F1 Curves': 'Requires Plotting Logic (Data is within metric_results)',
        'Confusion Matrix': 'N/A (Requires Specialized Implementation)'
    }

    for k, v in results.items():
        print(f"{k}: {v}")

    print("Evaluation complete.")

    return results


# --- Main Training Function (ADJUSTED DATALOADER COLLATE FN) ---

def train_detr_model(
        epochs=100,
        batch_size=8,
        learning_rate=1e-4,
        backbone_lr_multiplier=0.1,
        patience=20  # Added patience parameter for early stopping
):
    """
    Initializes and trains a DETR model on the prepared basketball dataset.
    """
    # --- 1. Setup Logging Directory ---
    project_name = 'DETR_Basketball_Detection'
    # Updated short name to match the new stable model
    model_name_short = "detr-resnet-50"

    # Create a unique run name based on model, epochs, batch size, and timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{model_name_short}_{epochs}e_{batch_size}b_{timestamp}"
    RUN_DIR = os.path.join("runs", "detr", run_name)
    os.makedirs(RUN_DIR, exist_ok=True)

    # Directory to save the best model found during validation
    BEST_MODEL_PATH = os.path.join(RUN_DIR, 'best_model')
    os.makedirs(BEST_MODEL_PATH, exist_ok=True)

    print(f"--- Starting DETR Training on Device: {DEVICE} ---")
    print(f"Results will be saved to: {RUN_DIR}")

    # 2. Load Processor and Model
    # SWITCHED to the standard, highly public DETR model to resolve the 401/404 error
    model_full_name = "facebook/detr-resnet-50"
    NUM_CLASSES = 9

    # --- IMAGE CONSISTENCY CHANGE HERE ---
    # Configuration to ensure image sizes match YOLO training (e.g., 640)
    processor = DetrImageProcessor.from_pretrained(
        model_full_name,
        size={"shortest_edge": IMAGE_SIZE, "longest_edge": IMAGE_SIZE}
    )

    model = DetrForObjectDetection.from_pretrained(
        model_full_name,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # 3. Load Data and Create DataLoaders
    train_image_base = os.path.join(BASE_DATA_DIR, 'train', 'images')
    val_image_base = os.path.join(BASE_DATA_DIR, 'val', 'images')
    test_image_base = os.path.join(BASE_DATA_DIR, 'test', 'images')

    train_samples = read_coco_json(TRAIN_JSON, train_image_base)
    val_samples = read_coco_json(VAL_JSON, val_image_base)
    test_samples = read_coco_json(TEST_JSON, test_image_base)

    train_dataset = DetrDataset(train_samples, processor)
    val_dataset = DetrDataset(val_samples, processor)
    test_dataset = DetrDataset(test_samples, processor)

    # --- ADJUSTMENT: Define collate_fn to use the processor for batching/padding ---

    # This is the correct way to use the processor's collate logic when
    # the processor itself is available globally (which it is here).
    # def detr_collate_fn_processor(batch):
    #     return processor.batch_decode(batch)

    # Since batch_decode is not directly accessible, we must use a function that
    # mimics the original `detr_collate_fn` but stacks only what we need, relying on
    # the processor in __getitem__ to handle resizing/padding implicitly.
    # The previous attempt to stack manually failed due to unequal sizes.
    # We must explicitly use the function provided by the processor: `pad_and_create_batch`

    # However, since we cannot guarantee the name of the internal function, we must
    # stick to the officially supported method of returning the list of dicts from
    # __getitem__ and letting the DataLoader handle the padding when processor's
    # collate function is used.

    # Since the previous approach caused both size error and AttributeError,
    # we must revert to a custom function that uses the processor's input/output format
    # correctly for padding.

    def detr_collate_fn_fixed(batch):
        # The batch is a list of dicts, where each dict contains pixel_values, mask, and labels.
        # We manually extract and pad the images and masks using a functional approach.

        pixel_values = [item['pixel_values'] for item in batch]
        pixel_mask = [item['pixel_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Determine max width and height in the batch
        max_h = max(p.shape[1] for p in pixel_values)
        max_w = max(p.shape[2] for p in pixel_values)

        padded_pixel_values = []
        padded_pixel_mask = []

        # Pad each image/mask to the max dimensions
        for img, mask in zip(pixel_values, pixel_mask):
            # Pad image tensor
            padding_h = max_h - img.shape[1]
            padding_w = max_w - img.shape[2]

            # Pad with 0s for the image (standard DETR padding)
            padded_img = torch.nn.functional.pad(img, (0, padding_w, 0, padding_h), value=0.0)
            padded_pixel_values.append(padded_img)

            # Pad mask tensor
            # Masks should be padded with 1s (indicating padding/no attention)
            padded_mask = torch.nn.functional.pad(mask, (0, padding_w, 0, padding_h), value=1.0)
            padded_pixel_mask.append(padded_mask)

        return {
            'pixel_values': torch.stack(padded_pixel_values),
            'pixel_mask': torch.stack(padded_pixel_mask),
            'labels': labels
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detr_collate_fn_fixed
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detr_collate_fn_fixed
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detr_collate_fn_fixed
    )

    print(f"Train Dataloader batches: {len(train_dataloader)}")
    print(f"Validation Dataloader batches: {len(val_dataloader)}")
    print(f"Test Dataloader batches: {len(test_dataloader)}")

    # 4. Setup Optimizer (Crucial for DETR)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": learning_rate * backbone_lr_multiplier},
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=learning_rate, weight_decay=1e-4)

    # --- Early Stopping Variables ---
    best_map = 0.0
    patience_counter = 0

    # 5. Training Loop
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} (Train)"):
            # DETR inputs are pixel_values, pixel_mask, and labels
            inputs = {
                'pixel_values': batch['pixel_values'].to(DEVICE),
                'pixel_mask': batch['pixel_mask'].to(DEVICE),
                # IMPORTANT: The model expects the processed tensor targets here for loss calculation.
                'labels': [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                           for t in batch['labels']]
            }

            optimizer.zero_grad()

            # The model internally handles the Hungarian matching and loss calculation
            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1} Train Loss: {avg_loss:.4f}")

        # --- Validation Step ---
        val_loss, val_mAP_50 = run_validation_epoch(model, val_dataloader, DEVICE, NUM_CLASSES)
        print(f"Epoch {epoch + 1} Val Loss: {val_loss:.4f} | Val mAP@50: {val_mAP_50 * 100:.2f}%")

        # --- Early Stopping Logic ---
        if val_mAP_50 > best_map:
            best_map = val_mAP_50
            patience_counter = 0
            # Save the model weights (This is the best model so far!)
            model.save_pretrained(BEST_MODEL_PATH)
            print(f"✅ New best model found and saved at {BEST_MODEL_PATH}. mAP@50: {best_map * 100:.2f}%")
        else:
            patience_counter += 1
            print(f"⚠️ mAP@50 did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement.")
            break

    print("\n--- DETR Training Complete! ---")

    # 6. Load Best Model for Final Test Evaluation
    try:
        if os.path.exists(BEST_MODEL_PATH):
            print(f"Loading best model from {BEST_MODEL_PATH} for final evaluation.")
            model = DetrForObjectDetection.from_pretrained(BEST_MODEL_PATH, num_labels=NUM_CLASSES).to(DEVICE)
        else:
            print("Warning: Best model path not found. Using current model for evaluation.")
    except Exception as e:
        print(f"Error loading best model: {e}. Using current model.")

    print("\n--- Running Final Test Evaluation ---")
    final_metrics = evaluate_model(model, test_dataloader, DEVICE, NUM_CLASSES)
    print(f"Final Test Metrics: {final_metrics}")

    # 7. Save Model Weights and Metrics
    try:
        # Save the FINAL metric results into the main RUN_DIR
        metrics_path = os.path.join(RUN_DIR, 'metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert metric values (like 50.25%) to simple floats (0.5025) for JSON
            json_metrics = {k: float(v.strip('%')) / 100 if isinstance(v, str) and v.endswith('%') else v for k, v in
                            final_metrics.items()}
            json.dump(json_metrics, f, indent=4)
        print(f"Final test metrics saved to: {metrics_path}")

        print(f"Best model weights are located in: {BEST_MODEL_PATH}")

    except Exception as e:
        print(f"Error during metrics saving: {e}")
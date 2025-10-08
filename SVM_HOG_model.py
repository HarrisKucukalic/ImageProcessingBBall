import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.validation import check_is_fitted
import seaborn as sns
import matplotlib.pyplot as plt

class SVM_HOG_ObjectDetector:
    """
    An object detector using Histogram of Oriented Gradients (HOG) and a Linear Support Vector Machine (SVM).

    This class is designed to train on images and labels provided in the YOLO format.
    It learns to distinguish between object patches (positives) and background patches (negatives).
    Detection is performed using a sliding window and non-maximum suppression.
    """

    def __init__(self, window_size=(64, 128), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Initialises the detector's parameters.

        Args:
            window_size (tuple): The (width, height) of the detection window. All image patches will be resized to this.
            orientations (int): Number of orientation bins for the HOG descriptor.
            pixels_per_cell (tuple): The size (in pixels) of a HOG cell.
            cells_per_block (tuple): The number of cells in each HOG block.
        """
        self.window_size = window_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

        # Initialise model components to None. They will be created during training.
        self.model = LinearSVC(C=1.0, max_iter=10000, dual="auto")
        self.scaler = StandardScaler()
        print("SVM-HOG Detector Initialised.")

    def _extract_hog_features(self, image):
        """Extracts HOG features from a single image patch."""
        # The image must be resized to the fixed window size
        resized_img = cv2.resize(image, self.window_size)
        # HOG works best on grayscale images
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        features = hog(gray_img, orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       transform_sqrt=True,
                       block_norm='L2-Hys',
                       feature_vector=True)
        return features

    def _load_yolo_data(self, image_dirs, label_dirs, num_neg_samples_per_image=10):
        """
        MODIFIED: Loads data from lists of image and label directories.
        """
        X, y = [], []

        # Iterate over all the provided directory pairs
        for image_dir, label_dir in zip(image_dirs, label_dirs):
            print(f"\nLoading data from {image_dir}")
            if not os.path.isdir(image_dir):
                print(f"DEBUG: Image directory not found at: {image_dir}")
                continue
            for filename in os.listdir(image_dir):
                try:
                    # Check if the file is a valid image
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    # Construct the full, guaranteed-to-exist image path
                    img_path = os.path.join(image_dir, filename)

                    # Derive the basename for the label file by splitting at the first dot
                    basename, extension = os.path.splitext(filename)
                    label_path = os.path.join(label_dir, basename + '.txt')

                    # Debugging
                    if not os.path.exists(label_path):
                        print(f"DEBUG: Image found '{img_path}', but label not found at '{label_path}'")
                        continue

                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"DEBUG: Failed to read image file: {img_path}")
                        continue

                    h, w, _ = image.shape
                    positive_boxes = []

                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if not lines:
                            continue

                        for line in lines:
                            parts = line.split()
                            if len(parts) < 5: continue
                            class_id, x_center, y_center, width, height = map(float, parts[:5])

                            x_min = int((x_center - width / 2) * w)
                            y_min = int((y_center - height / 2) * h)
                            x_max = int((x_center + width / 2) * w)
                            y_max = int((y_center + height / 2) * h)

                            positive_boxes.append([x_min, y_min, x_max, y_max])

                            patch = image[y_min:y_max, x_min:x_max]
                            if patch.size > 0:
                                features = self._extract_hog_features(patch)
                                X.append(features)
                                y.append(1)

                    if positive_boxes:
                        neg_count = 0
                        max_attempts = num_neg_samples_per_image * 20
                        attempts = 0
                        while neg_count < num_neg_samples_per_image and attempts < max_attempts:
                            attempts += 1
                            rand_x = random.randint(0, w - self.window_size[0]) if w > self.window_size[0] else 0
                            rand_y = random.randint(0, h - self.window_size[1]) if h > self.window_size[1] else 0

                            is_overlap = any(
                                not (rand_x + self.window_size[0] < px_min or rand_x > px_max or \
                                     rand_y + self.window_size[1] < py_min or rand_y > py_max)
                                for px_min, py_min, px_max, py_max in positive_boxes
                            )

                            if not is_overlap:
                                patch = image[rand_y:rand_y + self.window_size[1], rand_x:rand_x + self.window_size[0]]
                                if patch.shape[0] == self.window_size[1] and patch.shape[1] == self.window_size[0]:
                                    features = self._extract_hog_features(patch)
                                    X.append(features)
                                    y.append(0)
                                    neg_count += 1
                except Exception as e:
                    print(f"ERROR: Could not process {filename}. Reason: {e}")

            return np.array(X), np.array(y)

    def train(self, image_dirs, label_dirs):
        """
        Trains the SVM model using lists of directories.

        Args:
            image_dirs (list): List of paths to directories containing images.
            label_dirs (list): List of paths to directories containing YOLO labels.
        """
        # Load data and extract HOG features from all specified directories
        X, y = self._load_yolo_data(image_dirs, label_dirs)

        if len(y) == 0:
            print("No data was loaded. Aborting training. Check your directory paths.")
            return

        print(f"Total data loaded: {len(y)} samples ({np.sum(y)} positive, {len(y) - np.sum(y)} negative).")

        # Fit the scaler and transform the data
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        # Train the SVM classifier
        print("Training SVM classifier...")
        self.model.fit(X_scaled, y)
        print("Training complete. ✅")

    def _non_max_suppression(self, boxes, overlap_thresh):
        """Non-Maximum Suppression implementation."""
        if len(boxes) == 0:
            return []

        pick = []
        x1 = boxes[:, 0].astype(float)
        y1 = boxes[:, 1].astype(float)
        x2 = boxes[:, 2].astype(float)
        y2 = boxes[:, 3].astype(float)
        scores = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        return boxes[pick][:, :4].astype(int).tolist()

    def detect(self, image_path, confidence_threshold=0.5, scale_factor=1.5, step_size=16):
        """
        Detects objects in a new image using a sliding window.

        Args:
            image_path (str): Path to the image to perform detection on.
            confidence_threshold (float): Minimum confidence score from the SVM to count as a detection.
            scale_factor (float): Factor to downscale the image at each pyramid level.
            step_size (int): The step size for the sliding window in pixels.

        Returns:
            list: A list of final bounding boxes [x_min, y_min, x_max, y_max].
        """
        detections = []
        image = cv2.imread(image_path)
        original_h, original_w, _ = image.shape

        # Image Pyramid
        current_scale = 1.0
        current_img = image.copy()

        while current_img.shape[0] >= self.window_size[1] and current_img.shape[1] >= self.window_size[0]:
            # Sliding Window
            for y in range(0, current_img.shape[0] - self.window_size[1] + 1, step_size):
                for x in range(0, current_img.shape[1] - self.window_size[0] + 1, step_size):
                    # Extract patch
                    patch = current_img[y:y + self.window_size[1], x:x + self.window_size[0]]

                    # Get features and classify
                    features = self._extract_hog_features(patch).reshape(1, -1)
                    features_scaled = self.scaler.transform(features)

                    # Use decision_function for a confidence score
                    confidence = self.model.decision_function(features_scaled)

                    if confidence > confidence_threshold:
                        # Map window back to original image coordinates
                        x_min = int(x / current_scale)
                        y_min = int(y / current_scale)
                        x_max = int((x + self.window_size[0]) / current_scale)
                        y_max = int((y + self.window_size[1]) / current_scale)
                        detections.append([x_min, y_min, x_max, y_max, confidence[0]])

            # Downscale the image for the next pyramid level
            new_width = int(current_img.shape[1] / scale_factor)
            new_height = int(current_img.shape[0] / scale_factor)
            current_img = cv2.resize(current_img, (new_width, new_height))
            current_scale *= (original_w / new_width)

        # Non-Maximum Suppression (NMS)
        final_boxes = self._non_max_suppression(np.array(detections), overlap_thresh=0.3)
        return final_boxes

    def evaluate_model(self, image_dirs, label_dirs):
        """
        Evaluates the trained SVM classifier on a given dataset.

        This method loads positive and negative patches, predicts their labels,
        and prints a detailed classification report and confusion matrix.

        Args:
            image_dirs (list): List of paths to directories containing images.
            label_dirs (list): List of paths to directories containing YOLO labels.
        """
        try:
            # Check if the model and scaler have been fitted
            check_is_fitted(self.model)
            check_is_fitted(self.scaler)
        except:
            print("Model is not yet trained. Please call the 'train' method first.")
            return

        print("Loading evaluation data...")
        X_eval, y_true = self._load_yolo_data(image_dirs, label_dirs, num_neg_samples_per_image=10)

        if X_eval.shape[0] == 0:
            print("No data loaded for evaluation. Please check paths.")
            return

        print(f"Evaluating on {len(y_true)} samples...")
        # Scale features using the already-fitted scaler
        X_eval_scaled = self.scaler.transform(X_eval)

        # Make predictions
        y_pred = self.model.predict(X_eval_scaled)

        # Generate performance reports
        print("\nClassification Report")
        # target_names: 0 is 'background', 1 is 'object'
        print(classification_report(y_true, y_pred, target_names=['background', 'object']))
        print("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        # Confusion Matrix Plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['background', 'object'],
                    yticklabels=['background', 'object'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

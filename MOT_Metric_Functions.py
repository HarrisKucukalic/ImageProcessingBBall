import numpy as np
import pandas as pd
import os
import cv2
from BasketballAnalyser import BasketballAnalyser
from MOT_SIFT import MOT_SIFT

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


def load_tracking_data(filepath):
    """Loads tracking data (ground truth or tracker output) into a dictionary.

    Args:
        filepath (str): Path to the tracking data file.

    Returns:
        dict: A dictionary where keys are frame numbers and values are lists of
              tuples, with each tuple being (id, [x1, y1, w, h]).
    """
    data = {}
    if not os.path.exists(filepath):
        return data
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            track_id = int(parts[1])
            box = [float(p) for p in parts[2:6]]  # [x1, y1, w, h]
            if frame not in data:
                data[frame] = []
            data[frame].append((track_id, box))
    return data


def iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) between two boxes."""
    # Convert from [x, y, w, h] to [x1, y1, x2, y2]
    boxA_coords = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
    boxB_coords = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]

    xA = max(boxA_coords[0], boxB_coords[0])
    yA = max(boxA_coords[1], boxB_coords[1])
    xB = min(boxA_coords[2], boxB_coords[2])
    yB = min(boxA_coords[3], boxB_coords[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val


def manual_evaluate_tracker(gt_path, ts_path, iou_threshold=0.5):
    """
    Manually calculates MOTA, HOTA, and IDF1 without external libraries.
    """
    gt_data = load_tracking_data(gt_path)
    ts_data = load_tracking_data(ts_path)

    if not gt_data:
        print(f"Warning: Ground truth file '{gt_path}' is missing or empty.")
        return {'MOTA': 'N/A', 'HOTA': 'N/A', 'IDF1': 'N/A'}
    if not ts_data:
        print(f"Warning: Tracker file '{ts_path}' is empty.")
        return {'MOTA': 0, 'HOTA': 0, 'IDF1': 0}

    # Initialise counters for metrics
    num_gt_objects = 0
    tp, fp, fn = 0, 0, 0
    id_switches = 0

    # HOTA accumulators
    hota_accum = 0.0

    # Track matching history for ID switches
    prev_matches = {}  # {ts_id: gt_id}

    # Association data for HOTA
    gt_to_ts_matches = {}  # {gt_id: ts_id}
    ts_to_gt_matches = {}  # {ts_id: gt_id}

    all_frames = sorted(list(set(gt_data.keys()) | set(ts_data.keys())))

    for frame in all_frames:
        gt_in_frame = gt_data.get(frame, [])
        ts_in_frame = ts_data.get(frame, [])
        num_gt_objects += len(gt_in_frame)

        if not gt_in_frame and not ts_in_frame:
            continue

        gt_ids = [g[0] for g in gt_in_frame]
        gt_boxes = [g[1] for g in gt_in_frame]
        ts_ids = [t[0] for t in ts_in_frame]
        ts_boxes = [t[1] for t in ts_in_frame]

        # Frame-level matching
        matches = []
        if gt_in_frame and ts_in_frame:
            iou_matrix = np.zeros((len(gt_boxes), len(ts_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, ts_box in enumerate(ts_boxes):
                    iou_matrix[i, j] = iou(gt_box, ts_box)

            # Use a greedy approach to find matches
            gt_matched = [False] * len(gt_boxes)
            ts_matched = [False] * len(ts_boxes)

            # Iterate through potential matches in order of highest IoU
            for _ in range(len(gt_boxes) * len(ts_boxes)):
                if np.max(iou_matrix) < iou_threshold:
                    break

                gt_idx, ts_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

                if not gt_matched[gt_idx] and not ts_matched[ts_idx]:
                    matches.append((gt_ids[gt_idx], ts_ids[ts_idx]))
                    gt_matched[gt_idx] = True
                    ts_matched[ts_idx] = True

                iou_matrix[gt_idx, ts_idx] = 0  # Mark as used

        # Update TP, FP, FN
        frame_tp = len(matches)
        frame_fn = len(gt_ids) - frame_tp
        frame_fp = len(ts_ids) - frame_tp

        tp += frame_tp
        fn += frame_fn
        fp += frame_fp

        # Update ID Switches
        current_matches_ts_gt = {ts_id: gt_id for gt_id, ts_id in matches}
        for ts_id, gt_id in current_matches_ts_gt.items():
            if ts_id in prev_matches and prev_matches[ts_id] != gt_id:
                id_switches += 1
        prev_matches = current_matches_ts_gt

        # Accumulate data for HOTA's Association score
        for gt_id, ts_id in matches:
            gt_to_ts_matches.setdefault(gt_id, []).append(ts_id)
            ts_to_gt_matches.setdefault(ts_id, []).append(gt_id)

    # Final Metric Calculations

    # MOTA
    mota = 1.0 - ((fn + fp + id_switches) / num_gt_objects) if num_gt_objects > 0 else 0.0

    # IDF1
    idf1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # HOTA
    if tp > 0:
        # Calculate Detection Accuracy (DetA)
        det_a = tp / (tp + fn + fp)

        # Calculate Association Accuracy (AssA)
        association_scores = []
        all_matched_gt_ids = set()
        for gt_list in gt_to_ts_matches.values():
            all_matched_gt_ids.update(gt_list)

        for gt_id, matched_ts_ids in gt_to_ts_matches.items():
            for ts_id in set(matched_ts_ids):
                # Count true positive associations (TPA)
                tpa = sum(1 for ts in matched_ts_ids if ts == ts_id)
                # Count false negative associations (FNA)
                fna = len(gt_data.get(gt_id, [])) - tpa
                # Count false positive associations (FPA)
                fpa = sum(1 for gt in ts_to_gt_matches.get(ts_id, []) if gt != gt_id)

                # Association IoU for this match
                ass_iou = tpa / (tpa + fna + fpa) if (tpa + fna + fpa) > 0 else 0
                association_scores.append(ass_iou)

        ass_a = np.mean(association_scores) if association_scores else 0.0

        # HOTA is the geometric mean of DetA and AssA
        hota = np.sqrt(det_a * ass_a)
    else:
        hota = 0.0

    return {'MOTA': mota, 'HOTA': hota, 'IDF1': idf1}


def run_tracker_and_save(model_path, video_source, tracker_config, output_file_path, start_time="0:00", end_time=None):
    """
    Runs the BasketballAnalyser with a specific tracker on a video segment and saves the results.
    Handles YouTube URLs and processes video between start_time and end_time.
    """
    tracker_name_to_print = tracker_config if tracker_config != 'custom' else 'MOT-SIFT'
    print(f"Running analysis for tracker: {tracker_name_to_print}")

    if tracker_config == 'custom':
        tracker_to_run = MOT_SIFT(
            detection_conf=0.4,
            sift_good_dist=500.0,
            min_sift_score=10.0,
            accumulate_sift=3,
            max_age=30
        )
    else:
        tracker_to_run = tracker_config

    analyser = BasketballAnalyser(
        model_path=model_path,
        video_source=video_source,
        tracker_config=tracker_to_run
    )

    video_url = video_source
    video_fps = 30
    if 'youtube.com' in video_source or 'youtu.be' in video_source:
        if yt_dlp is None:
            print("\nError: A YouTube URL was provided, but 'yt-dlp' is not installed.")
            print("Please install it by running: pip install yt-dlp")
            return
        print(f"\nYouTube URL detected for '{video_source}'. Extracting direct video link...")
        try:
            ydl_opts = {'format': 'best[ext=mp4][height<=720]', 'noplaylist': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_source, download=False)
                video_url = info['url']
                video_fps = info.get('fps', 30)
                print("Direct link extracted successfully.")
        except Exception as e:
            print(f"Error extracting YouTube URL: {e}")
            return

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_url}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0: video_fps = fps

    start_frame = 0
    end_frame = float('inf')
    try:
        s_minutes, s_seconds = map(int, start_time.split(':'))
        start_total_seconds = (s_minutes * 60) + s_seconds
        start_frame = int(start_total_seconds * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    except (ValueError, IndexError):
        print(f"Invalid start time format: '{start_time}'. Starting from beginning.")

    if end_time:
        try:
            e_minutes, e_seconds = map(int, end_time.split(':'))
            end_total_seconds = (e_minutes * 60) + e_seconds
            end_frame = int(end_total_seconds * video_fps)
            print(f"Processing video from {start_time} to {end_time} (Frames {start_frame} to {end_frame}).")
        except (ValueError, IndexError):
            print(f"Invalid end time format: '{end_time}'. Processing to the end.")
    else:
        print(f"Processing video from {start_time} to end (Frame {start_frame} onwards).")

    frame_number = start_frame
    with open(output_file_path, 'w') as f_out:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_number >= end_frame:
                break

            print(f"Processing frame {frame_number}...", end='\r')

            if isinstance(analyser.tracker_config, str):
                player_results = analyser.model.track(
                    source=frame, conf=analyser.conf_thresh, iou=analyser.iou_thresh,
                    tracker=analyser.tracker_config, classes=[3], persist=True, verbose=False
                )[0]
                player_tracks = player_results.boxes.data.cpu().numpy() if player_results.boxes.id is not None else np.empty(
                    (0, 7))
            else:
                player_tracks = analyser.tracker_config.process_frame(frame, frame_number)

            for track in player_tracks:
                x1, y1, x2, y2, track_id, conf, _ = track
                w = x2 - x1
                h = y2 - y1
                line = f"{frame_number},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
                f_out.write(line)

            frame_number += 1

    cap.release()
    print(f"\nFinished processing. Results saved to '{output_file_path}'")


def calculate_track_stability_metrics(ts_path):
    """
    Calculates the mean and variance of the number of active tracks per frame.
    """
    try:
        df = pd.read_csv(ts_path, header=None, usecols=[0, 1], names=['frame_id', 'track_id'])
        if df.empty:
            return 0, 0
        track_counts_per_frame = df.groupby('frame_id')['track_id'].nunique()
        mean_tracks = track_counts_per_frame.mean()
        variance_tracks = track_counts_per_frame.var()
        return mean_tracks, variance_tracks
    except Exception as e:
        print(f"Could not calculate stability for {ts_path}: {e}")
        return 0, 0


if __name__ == "__main__":
    # Use the best performing object detector - YOLOv12n
    MODEL_PATH = "Basketball_Detection/yolov12n.pt_200_epochs_64_batch_size_augmented/weights/best.pt"
    VIDEO_SOURCE = "Q4_side_480-510.mp4"
    GT_FILE_PATH = "data/gt/gt.txt"

    # Time segment for the 30-second video clip with ground truth tracking data
    START_TIME = "0:00"
    END_TIME = "0:30"

    OUTPUT_DIR = "data/trackers_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    TRACKER_CONFIGS = {
        'ByteTrack': 'bytetrack.yaml',
        'BoT-SORT': 'botsort.yaml',
        'MOT-SIFT': 'custom'
    }

    TRACKER_FILES = {name: os.path.join(OUTPUT_DIR, f"{name.lower()}.txt") for name in TRACKER_CONFIGS}

    # Run Trackers to generate output CSVs
    # print("Generating Tracker Output Files")
    # for tracker_name, tracker_config in TRACKER_CONFIGS.items():
    #     output_path = TRACKER_FILES[tracker_name]
    #     run_tracker_and_save(MODEL_PATH, VIDEO_SOURCE, tracker_config, output_path, START_TIME, END_TIME)

    # Run evaluations on the outputted CSVs
    print("\nEvaluating Tracker Performance")
    all_results = {}
    for tracker_name, tracker_path in TRACKER_FILES.items():
        if not os.path.exists(tracker_path):
            print(f"Warning: Tracker file for '{tracker_name}' not found. Skipping.")
            continue

        print(f"Evaluating {tracker_name}...")
        metrics = manual_evaluate_tracker(GT_FILE_PATH, tracker_path)

        mean_tracks, variance_tracks = calculate_track_stability_metrics(tracker_path)
        metrics['Avg Tracks'] = mean_tracks
        metrics['Track Variance'] = variance_tracks

        all_results[tracker_name] = metrics
        print(f"Finished evaluating {tracker_name}.")

    # Print MOTA, HOTA and IDF1
    if all_results:
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        pd.options.display.float_format = '{:.3f}'.format
        print("\n--- MOT Metrics Comparison ---")
        print(results_df.to_string())
        print("----------------------------\n")
    else:
        print("\nNo trackers were evaluated. Please check paths and ensure tracking completed successfully.")


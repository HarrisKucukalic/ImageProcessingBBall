from ultralytics import YOLO
from StatsRecorder import *
import yt_dlp

CLASS_COLORS = {
    'player': (0, 255, 0),       # Green
    'basketball': (255, 165, 0),  # Orange
    'rim': (0, 0, 255)            # Red
}
# Helper function for RE-ID
def get_bbox_center(bbox):
    """Calculates the center of a bounding box."""
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

class BasketballAnalyser:
    """
    Main class to orchestrate the detection, tracking, and analysis process.
    """

    def __init__(self, model_path, video_source, tracker_config='custom_MOT_tracker.yaml.yaml', conf_thresh=0.5, iou_thresh=0.7):
        # Initialise YOLO model
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        # Store video source and config
        self.video_source = video_source
        self.tracker_config = tracker_config
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # Initialise statistics recorder
        self.stats_recorder = None
        self.frame_number = 0
        # How close (in pixels) a new detection must be to a lost player to be re-linked
        self.reid_distance_threshold = 150
        # How many frames a player can be lost before we consider re-linking them
        self.reid_frame_threshold = 60

    def _draw_tracks(self, frame, tracks, player_stats):
        """Draws dots and labels only for players and the ball."""
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, conf = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id, class_id = int(track_id), int(class_id)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            class_name = CLASS_NAMES.get(class_id, 'unknown')
            color = CLASS_COLORS.get(class_name, (255, 255, 255))

            # Only draw dots for players and the ball
            if class_id == PLAYER_CLASS_ID or class_id == BALL_CLASS_ID:
                cv2.circle(frame, (center_x, center_y), 5, color, -1)

            # Only draw a text label if the object is a player
            if class_id == PLAYER_CLASS_ID:
                # Get the score for the current player from the stats object
                score = player_stats[track_id].score if track_id in player_stats else 0
                label = f"P-{track_id} S:{score}"
                # Adjust position and font size for the new, longer label
                cv2.putText(frame, label, (center_x - 25, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def process_video(self):
        """
        Processes the video source, performs object tracking, and displays the results.
        """
        video_url = self.video_source
        video_fps = 0
        if 'youtube.com' in self.video_source or 'youtu.be' in self.video_source:
            if yt_dlp is None:
                print("\nError: yt_dlp is not installed. Please run 'pip install yt_dlp' to process YouTube videos.\n")
                return
            try:
                print("YouTube URL detected, extracting direct video link...")
                ydl_opts = {'format': 'best[ext=mp4][height<=1080]'}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.video_source, download=False)
                    video_url = info['url']
                    video_fps = info.get('fps', 0)
            except Exception as e:
                print(f"Error extracting YouTube URL: {e}")
                return

        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            print(f"OpenCV Error: Could not open video source: {video_url}")
            return

        # If FPS wasn't retrieved from metadata (e.g., not a YouTube video), get it from OpenCV
        if video_fps == 0:
            video_fps = cap.get(cv2.CAP_PROP_FPS)

        # If FPS is still 0, default to 30
        if video_fps == 0:
            print("Warning: Could not determine video FPS. Defaulting to 30.")
            video_fps = 30

        frame_duration = 1 / video_fps
        self.stats_recorder = StatsRecorder(video_fps)

        # Create a resizable window
        cv2.namedWindow("Basketball Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Basketball Analysis", 1280, 720)  # Set a larger default size

        start_time = time.time()
        frame_count = 0

        try:
            while cap.isOpened():
                self.frame_number += 1
                loop_start = time.time()

                ret, frame = cap.read()
                if not ret: break

                annotated_frame = frame.copy()

                results_list = self.model.track(source=frame, conf=self.conf_thresh, iou=self.iou_thresh,
                                                tracker=self.tracker_config, persist=True, verbose=False)
                results = results_list[0]

                if results.boxes.id is not None:
                    detections_with_ids = results.boxes.data.cpu().numpy()

                    # Custom RE-ID Logic ---

                    # Update memory for all currently tracked roster players
                    roster_ids = set(self.stats_recorder.player_stats.keys())
                    for d in detections_with_ids:
                        track_id = int(d[4])
                        if track_id in roster_ids:
                            self.stats_recorder.player_stats[track_id].last_seen_frame = self.frame_number
                            self.stats_recorder.player_stats[track_id].last_bbox = d[0:4]

                    # Identify new players and recently lost roster players
                    current_player_detections = {int(d[4]): d for d in detections_with_ids if
                                                 int(d[6]) == PLAYER_CLASS_ID}
                    current_player_ids = set(current_player_detections.keys())

                    new_unrostered_ids = {pid for pid in current_player_ids if pid not in roster_ids}
                    lost_roster_ids = {rid for rid in roster_ids if rid not in current_player_ids}

                    # Attempt to re-link lost players
                    if new_unrostered_ids and lost_roster_ids:
                        ids_to_relink = {}
                        for new_id in new_unrostered_ids:
                            new_bbox = current_player_detections[new_id][0:4]
                            new_center = get_bbox_center(new_bbox)

                            best_match_id = -1
                            min_dist = float('inf')

                            for lost_id in lost_roster_ids:
                                player_stat = self.stats_recorder.player_stats[lost_id]
                                # Check if the player was lost recently
                                if self.frame_number - player_stat.last_seen_frame < self.reid_frame_threshold:
                                    lost_center = get_bbox_center(player_stat.last_bbox)
                                    dist = np.linalg.norm(np.array(new_center) - np.array(lost_center))

                                    if dist < self.reid_distance_threshold and dist < min_dist:
                                        min_dist = dist
                                        best_match_id = lost_id

                            if best_match_id != -1:
                                ids_to_relink[new_id] = best_match_id

                        # Apply the re-links to the main detection list
                        for i in range(len(detections_with_ids)):
                            track_id = int(detections_with_ids[i][4])
                            if track_id in ids_to_relink:
                                detections_with_ids[i][4] = ids_to_relink[track_id]  # Re-assign the ID

                    # Filter players to maintain a roster of 10 on the court
                    # (This logic runs after re-linking to ensure roster stability)
                    known_player_ids = set(self.stats_recorder.player_stats.keys())
                    player_detections = [d for d in detections_with_ids if int(d[6]) == PLAYER_CLASS_ID]
                    other_detections = [d for d in detections_with_ids if int(d[6]) != PLAYER_CLASS_ID]

                    player_detections.sort(key=lambda x: x[5], reverse=True)

                    known_players_in_frame = [p for p in player_detections if int(p[4]) in known_player_ids]
                    new_players_in_frame = [p for p in player_detections if int(p[4]) not in known_player_ids]

                    available_roster_slots = 10 - len(known_player_ids)
                    players_to_add = new_players_in_frame[:max(0, available_roster_slots)]
                    final_detections = known_players_in_frame + players_to_add + other_detections
                    tracks_for_stats = [((d[0:4]), int(d[4]), int(d[6])) for d in final_detections]
                    self.stats_recorder.update(tracks_for_stats)

                    tracks_for_drawing = [(d[0], d[1], d[2], d[3], d[4], d[6], d[5]) for d in final_detections]
                    annotated_frame = self._draw_tracks(annotated_frame, tracks_for_drawing,
                                                        self.stats_recorder.player_stats)

                annotated_frame = self.stats_recorder.draw_stats(annotated_frame)

                # Display FPS
                elapsed_time = time.time() - start_time
                fps = self.frame_number / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(annotated_frame, f"Processing FPS: {fps:.2f}", (20, annotated_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Basketball Analysis", annotated_frame)

                processing_time = time.time() - loop_start
                wait_time_ms = int(max(1, (frame_duration - processing_time) * 1000))

                if cv2.waitKey(wait_time_ms) & 0xFF == ord("q"):
                    print("'q' pressed, stopping.")
                    break

        except Exception as e:
            print(f"An error occurred during video processing: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Processing finished.")

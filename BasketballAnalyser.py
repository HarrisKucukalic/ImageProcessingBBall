from ultralytics import YOLO
from StatsRecorder import StatsRecorder
from Team import Team
import cv2
import numpy as np
from collections import deque
from PIL import Image  # Still useful for general image manipulation, but no longer for OCR
import cvzone  # For utils functions
import math  # For math calculations
# Import external shot detection utilities
from shot_utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device

# Try to import yt_dlp, if error, set to None
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Centralised definitions for class IDs
PLAYER_CLASS_ID = 3
BALL_CLASS_ID = 0
HOOP_CLASS_ID = 1


#  Helper function to find centre of box
def get_bbox_centre(bbox):
    """Calculates the centre of a bounding box."""
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


class BasketballAnalyser:
    """
    Main class to orchestrate the detection, tracking, and team-based analysis.
    Now supports real-time display on local PCs via OpenCV windows.
    """

    def __init__(self, model_path, video_source, tracker_config='botsort.yaml', conf_thresh=0.5, iou_thresh=0.7,
                 start_time="0:00", output_video_path=None):  # Added output_video_path for optional recording

        # --- PRIMARY MODEL (Tracks Players) ---
        self.model = YOLO(model_path)

        # --- SECONDARY MODEL (Detects Ball/Hoop using specialized weights) ---
        try:
            self.shot_model = YOLO("shot_detector_external.pt")
            print("Successfully loaded specialized shot model 'shot_detector_external.pt'.")
        except Exception as e:
            # Fallback to using the primary model for all detections if specialized model fails to load
            print(
                f"Warning: Could not load specialized shot model 'shot_detector_external.pt'. Falling back to primary model for ball/hoop. Error: {e}")
            self.shot_model = self.model

        self.video_source = video_source
        self.tracker_config = tracker_config
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.stats_recorder = None
        self.frame_number = 0
        self.start_time = start_time
        self.teams_initialised = False
        # The threshold for determining if a jersey is light or dark (0-255)
        self.lightness_threshold = 130
        # Store a short history of recent, valid ball positions for custom tracker logic
        self.ball_position_history = deque(maxlen=4)
        # The maximum distance (in pixels) the ball can travel between frames
        self.max_ball_movement = 50
        self.homography_matrix = None
        self.court_template = None
        team_a_colour = (0, 255, 0)  # Green
        team_b_colour = (0, 0, 255)  # Red
        self.team_a = Team('A', team_a_colour)
        self.team_b = Team('B', team_b_colour)
        self.max_players = 10
        self.fixed_ids = deque(range(1, self.max_players + 1))
        self.tracker_to_fixed_id = {}
        # Threshold for ball and hoop proximity check
        self.ball_hoop_proximity_threshold = 30

        # Output video file logic
        self.output_video_path = output_video_path
        self.video_writer = None

        # New ball re-check logic:
        self.last_ball_detection_frame = 0
        # Default interval (5 seconds @ 30 FPS), will be updated with actual FPS later
        self.ball_recheck_interval = 150

        # --- SHOT DETECTION STATE (using external logic's requirements) ---
        self.ball_pos_shot = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos_shot = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.shot_up = False  # Ball has reached the 'up' phase of a shot
        self.shot_down = False  # Ball has reached the 'down' phase of a shot
        self.up_frame = 0
        self.down_frame = 0
        self.shot_overlay_text = "Waiting..."
        self.shot_overlay_color = (0, 0, 0)
        self.fade_frames = 30  # Duration of the result fade (30 frames = 1 second at 30 FPS)
        self.fade_counter = 0
        self.device = get_device()  # Get CUDA/CPU device from shot_utils

    def _analyse_jersey_properties(self, frame, bbox):
        """
        Determines team assignment by analysing lightness.
        Note: OCR components removed, returns None for jersey crop.
        """
        x1, y1, x2, y2 = map(int, bbox)
        if x1 >= x2 or y1 >= y2:
            return None, None

        # Define a tight region for the torso/jersey (middle 40% horizontally, upper-middle vertically)
        box_width = x2 - x1
        box_height = y2 - y1
        torso_x1 = x1 + int(box_width * 0.25)
        torso_x2 = x1 + int(box_width * 0.75)
        torso_y1 = y1 + int(box_height * 0.3)
        torso_y2 = y1 + int(box_height * 0.65)  # Avoid shorts/lower body

        # Ensure crop coordinates are within frame bounds
        h, w, _ = frame.shape
        torso_x1 = max(0, torso_x1)
        torso_x2 = min(w, torso_x2)
        torso_y1 = max(0, torso_y1)
        torso_y2 = min(h, torso_y2)

        # Crop the frame to this torso region for lightness check
        jersey_img_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]

        if jersey_img_crop.size == 0:
            return None, None

        # 1. Team Assignment (Lightness Check)
        gray_img = cv2.cvtColor(jersey_img_crop, cv2.COLOR_BGR2GRAY)
        average_intensity = np.mean(gray_img)

        if average_intensity > self.lightness_threshold:
            team_id = self.team_a.team_id  # Team A is 'Light'
        else:
            team_id = self.team_b.team_id  # Team B is 'Dark'

        # Return team assignment and None for the removed jersey crop
        return team_id, None

    def _track_ball(self, detections):
        """
        Refines ball tracking by removing outliers and interpolating short gaps.
        Now includes logic to force re-detection periodically for re-acquisition.
        """
        # Find the highest confidence ball detection in the current frame
        ball_detections = [d for d in detections if int(d[6]) == BALL_CLASS_ID]
        current_ball_centre = None

        if ball_detections:
            best_ball = max(ball_detections, key=lambda x: x[5])  # Index 5 is confidence
            current_ball_centre = get_bbox_centre(best_ball[0:4])

        # --- Re-acquisition Logic (Highest Priority) ---
        is_recheck_frame = (self.frame_number - self.last_ball_detection_frame) > self.ball_recheck_interval

        if is_recheck_frame and current_ball_centre is not None:
            # If we are due for a re-check AND a ball is detected, we accept it unconditionally.
            # Clear history to prevent distance check bias and force a clean acquisition.
            self.ball_position_history.clear()
            print(f"Ball re-acquisition forced at frame {self.frame_number}. History reset.")

        # --- Standard Outlier Rejection ---
        # This only runs if we have a new position AND we have a history to compare it to,
        # and it's NOT a forced re-acquisition frame (handled above).
        elif current_ball_centre and self.ball_position_history:
            last_known_pos = self.ball_position_history[-1]
            distance = np.linalg.norm(np.array(current_ball_centre) - np.array(last_known_pos))

            # If the ball has moved an impossibly large distance, treat it as a miss-detection
            if distance > self.max_ball_movement:
                current_ball_centre = None  # Discard the outlier due to too fast movement

        # Interpolation: If no valid ball is found, predict its position
        if not current_ball_centre and len(self.ball_position_history) >= 2:
            # Simple linear extrapolation
            last_pos = self.ball_position_history[-1]
            prev_pos = self.ball_position_history[-2]
            velocity = (np.array(last_pos) - np.array(prev_pos))
            # Predict the next position based on the last known velocity
            predicted_pos = tuple(map(int, np.array(last_pos) + velocity))
            current_ball_centre = predicted_pos

        # Update History
        if current_ball_centre:
            self.ball_position_history.append(current_ball_centre)
        # Else: Do nothing. History preserves the last valid track for interpolation/re-acquisition.

        return current_ball_centre

    def _process_shot_detections(self, detections):
        """
        Extracts ball and hoop positions from the current frame in the format
        required by shot_utils: (center_x, center_y), frame_count, width, height, conf.
        """
        ball_detections = [d for d in detections if int(d[6]) == BALL_CLASS_ID]
        hoop_detections = [d for d in detections if int(d[6]) == HOOP_CLASS_ID]

        # 1. Update Ball Position History (for shot logic)
        if ball_detections:
            best_ball = max(ball_detections, key=lambda x: x[5])  # Highest confidence ball
            x1, y1, x2, y2 = map(int, best_ball[0:4])
            w, h = x2 - x1, y2 - y1
            center = get_bbox_centre(best_ball[0:4])
            conf = math.ceil((best_ball[5] * 100)) / 100

            # The shot detector uses a soft check for low confidence near the hoop,
            # but we will rely on the detection confidence directly here to add the point.
            if conf > self.conf_thresh:  # Use the analyser's confidence threshold
                self.ball_pos_shot.append((center, self.frame_number, w, h, conf))

        # 2. Update Hoop Position History (for shot logic)
        if hoop_detections:
            best_hoop = max(hoop_detections, key=lambda x: x[5])  # Highest confidence hoop
            x1, y1, x2, y2 = map(int, best_hoop[0:4])
            w, h = x2 - x1, y2 - y1
            center = get_bbox_centre(best_hoop[0:4])
            conf = math.ceil((best_hoop[5] * 100)) / 100

            if conf > 0.5:  # Use a reasonable fixed confidence for the static hoop
                self.hoop_pos_shot.append((center, self.frame_number, w, h, conf))

        # 3. Clean Motion using external utilities
        self.ball_pos_shot = clean_ball_pos(self.ball_pos_shot, self.frame_number)
        if len(self.hoop_pos_shot) > 1:
            self.hoop_pos_shot = clean_hoop_pos(self.hoop_pos_shot)

    def _run_shot_logic(self):
        """
        Implements the state machine for shot detection using the cleaned ball and hoop data.
        Updates self.stats_recorder.makes and attempts.
        """
        # --- Decrement Fade Counter ---
        if self.fade_counter > 0:
            self.fade_counter -= 1
        else:
            # Reset text to neutral state once fade is complete, preventing the score from being drawn here.
            if self.shot_overlay_text != "Waiting...":
                self.shot_overlay_text = "Waiting..."
                self.shot_overlay_color = (0, 0, 0)  # Black/Neutral

        # --- Shot Detection Logic ---
        if len(self.hoop_pos_shot) > 0 and len(self.ball_pos_shot) > 0:
            # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
            if not self.shot_up:
                self.shot_up = detect_up(self.ball_pos_shot, self.hoop_pos_shot)
                if self.shot_up:
                    self.up_frame = self.ball_pos_shot[-1][1]

            if self.shot_up and not self.shot_down:
                self.shot_down = detect_down(self.ball_pos_shot, self.hoop_pos_shot)
                if self.shot_down:
                    self.down_frame = self.ball_pos_shot[-1][1]

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            # The original code checked every 10 frames (self.frame_count % 10 == 0). We'll keep that idea for throttling.
            if self.frame_number % 10 == 0:
                if self.shot_up and self.shot_down and self.up_frame < self.down_frame:

                    # 1. Update Stats Recorder
                    self.stats_recorder.attempts += 1
                    print(f"Shot Attempt Detected! Total attempts: {self.stats_recorder.attempts}")

                    # 2. Check for Score (using external utility)
                    if score(self.ball_pos_shot, self.hoop_pos_shot):
                        self.stats_recorder.makes += 1

                        # Assuming the player who had possession made the shot
                        player_id = self.stats_recorder.player_with_ball
                        if player_id is not None and player_id in self.stats_recorder.player_stats:
                            # Update team score via the player's team
                            team_id = self.stats_recorder.player_stats[player_id].team_id
                            team = self.stats_recorder.teams.get(team_id)
                            if team:
                                team.update_score()  # Update team score
                            self.stats_recorder.player_stats[player_id].update_points(2)  # Update individual points

                        self.shot_overlay_color = (0, 255, 0)  # Green for make (BGR)
                        self.shot_overlay_text = "MAKE!"
                        print("SHOT MADE!")

                    else:
                        self.shot_overlay_color = (0, 0, 255)  # Red for miss (BGR)
                        self.shot_overlay_text = "MISS"
                        print("SHOT MISSED.")

                    # 3. Start Visual Fade
                    self.fade_counter = self.fade_frames

                    # 4. Reset Shot State
                    self.shot_up = False
                    self.shot_down = False
                    self.up_frame = 0
                    self.down_frame = 0

    def _check_ball_hoop_proximity(self, detections):
        """Checks if the ball and hoop objects are close to each other."""
        # This is now largely redundant as shot logic is primary, suppressing print
        pass

    def _setup_birds_eye_view(self, frame_shape):
        """Creates the court template with a detailed outline and calculates the homography matrix."""
        h, w, _ = frame_shape
        # Define the dimensions and colours for the top-down court view (swapped for horizontal)
        court_w, court_h = 800, 470
        court_colour = (58, 112, 62)  # Green
        line_colour = (255, 255, 255)  # White
        self.court_template = np.zeros((court_h, court_w, 3), dtype=np.uint8)
        self.court_template[:] = court_colour

        # Centre circle
        cv2.circle(self.court_template, (court_w // 2, court_h // 2), 50, line_colour, 2)
        # Half-court line
        cv2.line(self.court_template, (court_w // 2, 0), (court_w // 2, court_h), line_colour, 2)

        # Top key
        key_width, key_height = 190, 160
        cv2.rectangle(self.court_template, (0, (court_h - key_height) // 2), (key_width, (court_h + key_height) // 2),
                      line_colour, 2)
        cv2.circle(self.court_template, (key_width, court_h // 2), 60, line_colour, 2)

        # Bottom key
        cv2.rectangle(self.court_template, (court_w - key_width, (court_h - key_height) // 2),
                      (court_w, (court_h + key_height) // 2), line_colour, 2)
        cv2.circle(self.court_template, (court_w - key_width, court_h // 2), 60, line_colour, 2)

        # Shift constant for the three-point arcs to move them further from the centre line
        ARC_SHIFT = -150

        # Three-point arcs (Shifted 50 pixels away from the center of the court)
        # Left side: key_width + ARC_SHIFT (moves centre right)
        cv2.ellipse(self.court_template, (key_width + ARC_SHIFT, court_h // 2), (235, 235), 0, -90, 90, line_colour, 2)
        # Right side: (court_w - key_width) - ARC_SHIFT (moves centre left)
        cv2.ellipse(self.court_template, (court_w - key_width - ARC_SHIFT, court_h // 2), (235, 235), 0, 90, 270,
                    line_colour, 2)

        # These are estimated points from a video frame (source)
        # and their corresponding locations on the top-down map (destination).
        src_points = np.array([[w * 0.3, h * 0.4], [w * 0.7, h * 0.4], [w * 0.9, h * 0.9], [w * 0.1, h * 0.9]],
                              dtype=np.float32)
        dst_points = np.array([[100, 100], [400, 100], [400, 400], [100, 400]], dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(src_points, dst_points)

    def _draw_birds_eye_view(self, current_player_ids):
        """Draws the top-down tactical view of player and ball positions."""
        tactical_view = self.court_template.copy()
        points_to_transform = []
        colours = []
        # Gather player positions and colours
        for pid in current_player_ids:
            stats = self.stats_recorder.player_stats.get(pid)
            if stats and stats.positions:
                points_to_transform.append(stats.positions[-1])
                colours.append(self.stats_recorder.teams[stats.team_id].primary_colour)
        # Add ball position if it exists
        if self.stats_recorder.ball_position:
            points_to_transform.append(self.stats_recorder.ball_position)
            colours.append((255, 165, 0))  # Orange for ball
        if points_to_transform:
            points_np = np.array(points_to_transform, dtype=np.float32).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(points_np, self.homography_matrix)
            for point, colour in zip(transformed_points, colours):
                x, y = int(point[0][0]), int(point[0][1])
                cv2.circle(tactical_view, (x, y), 5, colour, -1)

        cv2.imshow("Tactical View", tactical_view)

    def _draw_tracks(self, frame, active_fixed_player_ids):
        """Draws dots and labels for currently tracked players, colour-coded by team and showing fixed ID."""

        # Draw player dots and IDs
        for player_id in active_fixed_player_ids:
            stats = self.stats_recorder.player_stats.get(player_id)
            text_to_display = f"P{player_id}"

            if stats and stats.team_id and stats.positions:
                team = self.stats_recorder.teams.get(stats.team_id)
                if team:
                    dot_colour = team.primary_colour
                    # Draw only the last, most recent position
                    centre_x, centre_y = stats.positions[-1]
                    cv2.circle(frame, (centre_x, centre_y), 7, dot_colour, -1)

                    # Position the text above the player dot
                    cv2.putText(frame, text_to_display, (centre_x + 10, centre_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw the refined ball position
        if self.stats_recorder.ball_position:
            ball_x, ball_y = map(int, self.stats_recorder.ball_position)
            cv2.circle(frame, (ball_x, ball_y), 7, (255, 165, 0), -1)

        # --- Draw Hoop for shot detection debugging/visualisation (optional, using cleaned pos) ---
        if len(self.hoop_pos_shot) > 0:
            hoop_center = self.hoop_pos_shot[-1][0]
            cv2.circle(frame, hoop_center, 5, (128, 128, 0), -1)  # Draw center of cleaned hoop pos

        # --- Draw Shot Result Overlay ---
        # Note: We only draw the transient "MAKE!" or "MISS" text here, not the continuous score.
        if self.fade_counter > 0 and self.shot_overlay_text != "Waiting...":
            # Calculate text size to position it at the right top corner
            (text_width, text_height), _ = cv2.getTextSize(self.shot_overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
            text_x = frame.shape[1] - text_width - 40  # Right alignment with some margin
            text_y = 100  # Top margin

            # Display overlay text
            cv2.putText(frame, self.shot_overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        self.shot_overlay_color, 6)

            # Gradually fade out color after shot
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            # Add a faint coloured overlay effect to the whole screen for flair
            frame = cv2.addWeighted(frame, 1 - alpha, np.full_like(frame, self.shot_overlay_color), alpha * 0.5, 0)

        return frame

    def _draw_stats_window(self):
        """Generates and draws the stats frame containing time and gravity score."""

        # Create a blank black image for the stats display
        stats_h, stats_w = 250, 400
        stats_frame = np.zeros((stats_h, stats_w, 3), dtype=np.uint8)

        # 1. Draw Game Timer
        # This now relies on the current_time_string property in StatsRecorder
        time_text = f"Game Time: {self.stats_recorder.current_time_string}"
        cv2.putText(stats_frame, time_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 2. Draw Shot Stats (New Section)
        shot_stats_text = f"Shots: {self.stats_recorder.makes} / {self.stats_recorder.attempts}"
        cv2.putText(stats_frame, shot_stats_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
                    cv2.LINE_AA)

        # 3. Draw Gravity Score and Player ID (Shifted Down)
        gravity_id = self.stats_recorder.highest_gravity_player_id
        pressure_player_text = f"P{gravity_id}"

        # Title
        cv2.putText(stats_frame, "Off-Ball Gravity", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 191, 255), 2,
                    cv2.LINE_AA)

        # Gravity Score
        score_text = f"Score: {self.stats_recorder.gravity_score:.2f}"
        cv2.putText(stats_frame, score_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Highest Pressure Player
        player_text = f"Pressure Player: {pressure_player_text}"
        cv2.putText(stats_frame, player_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Game Stats", stats_frame)

    def _calculate_and_update_gravity(self, current_player_ids):
        """Calculates gravity exerted by each defender and identifies the one with the highest pressure."""
        player_with_ball_id = self.stats_recorder.player_with_ball
        # Reset high gravity player each frame
        self.stats_recorder.highest_gravity_player_id = None
        if not player_with_ball_id or player_with_ball_id not in self.stats_recorder.player_stats:
            self.stats_recorder.gravity_score = 0.0
            return

        ball_handler_stats = self.stats_recorder.player_stats[player_with_ball_id]
        if not ball_handler_stats.positions:
            self.stats_recorder.gravity_score = 0.0
            return

        ball_handler_pos = np.array(ball_handler_stats.positions[-1])
        ball_handler_team = ball_handler_stats.team_id

        total_gravity = 0.0
        max_individual_gravity = -1.0
        highest_gravity_player_id = None
        scaling_factor = 100.0

        for pid in current_player_ids:
            player_stats = self.stats_recorder.player_stats.get(pid)
            if player_stats and player_stats.team_id != ball_handler_team and player_stats.positions:
                opponent_pos = np.array(player_stats.positions[-1])
                distance = np.linalg.norm(ball_handler_pos - opponent_pos)

                individual_gravity = np.exp(-distance / scaling_factor)
                total_gravity += individual_gravity

                if individual_gravity > max_individual_gravity:
                    max_individual_gravity = individual_gravity
                    highest_gravity_player_id = pid

        self.stats_recorder.gravity_score = total_gravity
        self.stats_recorder.highest_gravity_player_id = highest_gravity_player_id

    def process_video(self):
        """Processes the video, identifies teams, tracks players, and writes output to file or displays in real-time."""
        original_video_source = self.video_source
        video_url = self.video_source
        video_fps = 0
        w, h = 0, 0  # Will be populated once video capture is opened

        if 'youtube.com' in self.video_source or 'youtu.be' in self.video_source:
            try:
                ydl_opts = {'format': 'best[ext=mp4][height<=1080]', 'noplaylist': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.video_source, download=False)
                    video_url, video_fps = info['url'], info.get('fps', 0)
            except Exception as e:
                print(f"Error extracting YouTube URL: {e}")
                print("Falling back to local video processing or a different video source.")
                video_url = original_video_source

        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened(): return print("Error: Could not open video source.")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if video_fps == 0: video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0: video_fps = 30

        # --- Set the re-check interval based on video FPS ---
        # 5 seconds is generally a safe re-check interval for ball tracking
        self.ball_recheck_interval = int(video_fps * 5)
        self.fade_frames = int(video_fps * 1)  # 1 second duration for fade
        print(f"Ball re-check interval set to {self.ball_recheck_interval} frames ({video_fps} FPS).")

        # Initialise VideoWriter if output_video_path is provided
        if self.output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
            self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, video_fps, (w, h))
            print(f"Video writer initialised. Output will be saved to: {self.output_video_path}")
        else:
            print("No output video path provided. Displaying results in real-time.")

        # --- Initialise OpenCV windows for local execution ---
        cv2.namedWindow("Basketball Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Basketball Analysis", 1280, 720)
        cv2.namedWindow("Tactical View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tactical View", 800, 470)
        cv2.namedWindow("Game Stats", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Game Stats", 400, 250)

        # Pass the initialised Team objects to the StatsRecorder
        self.stats_recorder = StatsRecorder(video_fps, self.team_a, self.team_b)

        if self.start_time != "0:00":
            try:
                parts = list(map(int, self.start_time.split(':')))
                minutes, seconds = (parts[0], parts[1]) if len(parts) == 2 else (0, 0)
                start_seconds = (minutes * 60) + seconds
                if start_seconds > 0:
                    start_frame = int(start_seconds * video_fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    self.frame_number = start_frame
            except ValueError:
                print("Invalid start time format. Defaulting to beginning.")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                self.frame_number += 1

                if self.homography_matrix is None:
                    # Setup birds eye view template and homography matrix
                    self._setup_birds_eye_view(frame.shape)

                # --- 1. DETECTION & TRACKING ---

                # 1A. Player Tracking (Primary model)
                player_results = self.model.track(
                    source=frame,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    tracker=self.tracker_config,
                    classes=[PLAYER_CLASS_ID],
                    persist=True,
                    verbose=False
                )[0]

                # Extract tracks (Full 7-element array: x1, y1, x2, y2, track_id, conf, class_id)
                player_tracks = player_results.boxes.data.cpu().numpy() if player_results.boxes.id is not None else np.empty(
                    (0, 7))

                # 1B. Ball/Hoop Detection (Specialized Shot model)
                shot_results = self.shot_model.predict(
                    source=frame,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    classes=[BALL_CLASS_ID, HOOP_CLASS_ID],
                    verbose=False
                )[0]

                # Extract detections (Standard 6-element array: x1, y1, x2, y2, conf, class_id)
                shot_detections_6d = shot_results.boxes.data.cpu().numpy() if len(shot_results.boxes) > 0 else np.empty(
                    (0, 6))

                # Reformat shot detections to match the 7-element tracked structure (x1, y1, x2, y2, track_id=-1, conf, class_id)
                shot_detections_7d = []
                for det in shot_detections_6d:
                    # Insert placeholder track_id (-1.0) at index 4
                    shot_detections_7d.append([det[0], det[1], det[2], det[3], -1.0, det[4], det[5]])

                # Combine tracks and detections into a single array
                current_detections = np.array(shot_detections_7d + list(player_tracks))
                current_player_detections = player_tracks

                # --- 2. ID MANAGEMENT ---
                current_tracker_ids = {int(d[4]) for d in current_player_detections}

                # Identify lost players and return their fixed IDs to the pool
                ids_to_remove = []
                for tracker_id, fixed_id in self.tracker_to_fixed_id.items():
                    if tracker_id not in current_tracker_ids:
                        self.fixed_ids.append(fixed_id)
                        ids_to_remove.append(tracker_id)
                        # No need to check self.fixed_id_to_jersey_number anymore

                for tracker_id in ids_to_remove:
                    fixed_id = self.tracker_to_fixed_id.get(tracker_id)
                    if fixed_id is not None:
                        del self.tracker_to_fixed_id[tracker_id]
                        self.stats_recorder.remove_player(fixed_id)

                # Assign fixed IDs to new players and perform team assignment
                for d in current_player_detections:
                    tracker_id = int(d[4])
                    bbox = d[0:4]

                    # Analyse jersey properties (team colour and get crop)
                    assigned_team, _ = self._analyse_jersey_properties(frame, bbox)  # No jersey_crop needed

                    # 1. Assign Fixed ID if new
                    if tracker_id not in self.tracker_to_fixed_id and self.fixed_ids:
                        fixed_id = self.fixed_ids.popleft()
                        self.tracker_to_fixed_id[tracker_id] = fixed_id

                        if assigned_team:
                            self.stats_recorder.add_player(fixed_id, assigned_team)

                    # Update stats for all currently detected players
                    fixed_id = self.tracker_to_fixed_id.get(tracker_id)
                    if fixed_id is not None:
                        centre_x = int((d[0] + d[2]) / 2)
                        centre_y = int((d[1] + d[3]) / 2)
                        player_stats = self.stats_recorder.player_stats.get(fixed_id)
                        if player_stats:
                            player_stats.update_position((centre_x, centre_y))

                # Now get the list of active fixed player IDs
                active_fixed_player_ids = list(self.stats_recorder.player_stats.keys())

                # --- 3. BALL TRACKING & SHOT DATA GATHERING ---
                # A. Update the primary ball position (used for possession/gravity/map)
                self.stats_recorder.ball_position = self._track_ball(current_detections)

                # B. Gather and clean data for robust SHOT DETECTION
                self._process_shot_detections(current_detections)

                # If a new ball position was successfully found (either via detection or interpolation),
                # update the last detection frame.
                if self.stats_recorder.ball_position is not None:
                    self.last_ball_detection_frame = self.frame_number

                # --- 4. ANALYTICS EXECUTION ---
                self._run_shot_logic()  # Must run before stats update to catch scores
                self.stats_recorder.update(current_detections, self.frame_number)
                self._calculate_and_update_gravity(active_fixed_player_ids)
                self._check_ball_hoop_proximity(current_detections)

                # --- 5. VISUALISATION ---
                annotated_frame = frame.copy()
                annotated_frame = self._draw_tracks(annotated_frame, active_fixed_player_ids)

                # --- Draw and Show Windows ---
                stats_frame_from_recorder = self.stats_recorder.get_stats_frame()
                cv2.imshow("Game Stats", stats_frame_from_recorder)  # Display the frame generated by the recorder

                self._draw_birds_eye_view(active_fixed_player_ids)
                cv2.imshow("Basketball Analysis", annotated_frame)

                # WRITE FRAME TO FILE
                if self.video_writer:
                    self.video_writer.write(annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"): break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cap.release()
            if self.video_writer:
                self.video_writer.release()
                print(f"Output video successfully saved to: {self.output_video_path}")
            # Ensure windows are closed properly
            cv2.destroyAllWindows()
            print("Processing finished.")

from ultralytics import YOLO
from StatsRecorder import StatsRecorder
from Team import Team
import cv2
import numpy as np
from collections import deque

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
    """

    def __init__(self, model_path, video_source, tracker_config='botsort.yaml', conf_thresh=0.5, iou_thresh=0.7,
                 start_time="0:00"):
        self.model = YOLO(model_path)
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
        # Store a short history of recent, valid ball positions
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

    def _get_player_team_assignment(self, frame, bbox):
        """
        Determines team assignment by analysing the average lightness of the jersey,
        focusing only on the torso to avoid bias from skin tone.
        """
        x1, y1, x2, y2 = map(int, bbox)
        if x1 >= x2 or y1 >= y2:
            return None
        # This focuses the analysis on the jersey and ignores head, arms, and legs.
        box_width = x2 - x1
        box_height = y2 - y1
        # Define a region for the torso (e.g., middle 50% horizontally, upper-middle vertically)
        torso_x1 = x1 + int(box_width * 0.35)
        torso_x2 = x1 + int(box_width * 0.65)
        torso_y1 = y1 + int(box_height * 0.35)
        torso_y2 = y1 + int(box_height * 0.45)  # Avoid shorts
        # Crop the frame to this torso region
        player_img = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if player_img.size == 0:
            return None
        # Convert the player image to grayscale to analyse lightness
        gray_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2GRAY)
        # Calculate the average pixel intensity of the torso
        average_intensity = np.mean(gray_img)
        # Assign to the 'Light' or 'Dark' team based on the threshold
        if average_intensity > self.lightness_threshold:
            return self.team_a.team_id  # Team A is 'Light'
        else:
            return self.team_b.team_id  # Team B is 'Dark'

    def _track_ball(self, detections):
        """
        Refines ball tracking by removing outliers and interpolating short gaps.
        This logic is adapted from the BallTracker class you provided.
        """
        # Find the highest confidence ball detection in the current frame
        ball_detections = [d for d in detections if int(d[6]) == BALL_CLASS_ID]
        current_ball_centre = None

        if ball_detections:
            best_ball = max(ball_detections, key=lambda x: x[5])  # Index 5 is confidence
            current_ball_centre = get_bbox_centre(best_ball[0:4])

        # Outlier Rejection: Check if the new position is plausible
        if current_ball_centre and self.ball_position_history:
            last_known_pos = self.ball_position_history[-1]
            distance = np.linalg.norm(np.array(current_ball_centre) - np.array(last_known_pos))
            # If the ball has moved an impossibly large distance, treat it as a miss-detection
            if distance > self.max_ball_movement:
                current_ball_centre = None  # Discard the outlier

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
        else:
            # If no ball is detected and interpolation is not possible, clear history
            self.ball_position_history.clear()

        return current_ball_centre

    def _check_ball_hoop_proximity(self, detections):
        """Checks if the ball and hoop objects are close to each other."""
        ball_detections = [d for d in detections if int(d[6]) == BALL_CLASS_ID]
        hoop_detections = [d for d in detections if int(d[6]) == HOOP_CLASS_ID]

        if ball_detections and hoop_detections:
            # Get the highest-confidence ball and hoop detections
            ball_bbox = max(ball_detections, key=lambda x: x[5])[0:4]
            hoop_bbox = max(hoop_detections, key=lambda x: x[5])[0:4]

            ball_center = get_bbox_centre(ball_bbox)
            hoop_center = get_bbox_centre(hoop_bbox)

            distance = np.linalg.norm(np.array(ball_center) - np.array(hoop_center))

            if distance < self.ball_hoop_proximity_threshold:
                print(f"Proximity Alert! Ball and hoop are close. Distance: {distance:.2f} pixels.")

    def _setup_birds_eye_view(self, frame_shape):
        """Creates the court template with a detailed outline and calculates the homography matrix."""
        h, w, _ = frame_shape
        # Define the dimensions and colours for the top-down court view (swapped for horizontal)
        court_w, court_h = 800, 470
        court_colour = (58, 112, 62)  # Green
        line_colour = (255, 255, 255)  # White
        self.court_template = np.zeros((court_h, court_w, 3), dtype=np.uint8)
        self.court_template[:] = court_colour

        # Center circle
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

        # Three-point arcs
        cv2.ellipse(self.court_template, (key_width, court_h // 2), (235, 235), 0, -90, 90, line_colour, 2)
        cv2.ellipse(self.court_template, (court_w - key_width, court_h // 2), (235, 235), 0, 90, 270, line_colour, 2)

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
        """Draws dots and labels for currently tracked players, colour-coded by team."""

        # Draw player dots and IDs
        for player_id in active_fixed_player_ids:
            stats = self.stats_recorder.player_stats.get(player_id)
            if stats and stats.team_id and stats.positions:
                team = self.stats_recorder.teams.get(stats.team_id)
                if team:
                    dot_colour = team.primary_colour
                    # Draw only the last, most recent position
                    centre_x, centre_y = stats.positions[-1]
                    cv2.circle(frame, (centre_x, centre_y), 7, dot_colour, -1)
                    # Add player ID label for debugging
                    cv2.putText(frame, str(player_id), (centre_x + 10, centre_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)
        # Draw the refined ball position
        if self.stats_recorder.ball_position:
            ball_x, ball_y = map(int, self.stats_recorder.ball_position)
            cv2.circle(frame, (ball_x, ball_y), 7, (255, 165, 0), -1)
        return frame

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
        """Processes the video, identifies teams, and tracks players."""
        original_video_source = self.video_source
        video_url = self.video_source
        video_fps = 0

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

        if video_fps == 0: video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0: video_fps = 30

        # Pass the initialised Team objects to the StatsRecorder
        self.stats_recorder = StatsRecorder(video_fps, self.team_a, self.team_b)
        cv2.namedWindow("Basketball Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Basketball Analysis", 1280, 720)
        cv2.namedWindow("Tactical View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tactical View", 800, 470)

        # New window for stats display
        cv2.namedWindow("Game Stats", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Game Stats", 400, 250)

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
                    self._setup_birds_eye_view(frame.shape)

                results = \
                    self.model.track(source=frame, conf=self.conf_thresh, iou=self.iou_thresh,
                                     tracker=self.tracker_config,
                                     persist=True, verbose=False)[0]

                # Get all player detections and their IDs from the current frame
                current_detections = results.boxes.data.cpu().numpy() if results.boxes.id is not None else []
                current_player_detections = [d for d in current_detections if int(d[6]) == PLAYER_CLASS_ID]

                # --- NEW ID MANAGEMENT LOGIC ---
                current_tracker_ids = {int(d[4]) for d in current_player_detections}

                # Identify lost players and return their fixed IDs to the pool
                ids_to_remove = []
                for tracker_id, fixed_id in self.tracker_to_fixed_id.items():
                    if tracker_id not in current_tracker_ids:
                        self.fixed_ids.append(fixed_id)
                        ids_to_remove.append(tracker_id)
                        print(f"Player with ID {fixed_id} lost. ID returned to pool.")
                for tracker_id in ids_to_remove:
                    fixed_id = self.tracker_to_fixed_id.get(tracker_id)
                    if fixed_id is not None:
                        del self.tracker_to_fixed_id[tracker_id]
                        self.stats_recorder.remove_player(fixed_id)

                # Assign fixed IDs to new players
                for d in current_player_detections:
                    tracker_id = int(d[4])
                    if tracker_id not in self.tracker_to_fixed_id and self.fixed_ids:
                        fixed_id = self.fixed_ids.popleft()
                        self.tracker_to_fixed_id[tracker_id] = fixed_id
                        assigned_team = self._get_player_team_assignment(frame, d[0:4])
                        if assigned_team:
                            self.stats_recorder.add_player(fixed_id, assigned_team)
                            print(f"Assigned new fixed ID {fixed_id} to tracker ID {tracker_id}. Team: {assigned_team}")

                # Update stats for all currently detected players
                for d in current_player_detections:
                    tracker_id = int(d[4])
                    fixed_id = self.tracker_to_fixed_id.get(tracker_id)
                    if fixed_id is not None:
                        centre_x = int((d[0] + d[2]) / 2)
                        centre_y = int((d[1] + d[3]) / 2)
                        player_stats = self.stats_recorder.player_stats.get(fixed_id)
                        if player_stats:
                            player_stats.update_position((centre_x, centre_y))

                # Now get the list of active fixed player IDs
                active_fixed_player_ids = list(self.stats_recorder.player_stats.keys())

                self.stats_recorder.ball_position = self._track_ball(current_detections)
                self.stats_recorder.update(current_detections, self.frame_number)
                self._calculate_and_update_gravity(active_fixed_player_ids)
                self._check_ball_hoop_proximity(current_detections)

                annotated_frame = frame.copy()
                annotated_frame = self._draw_tracks(annotated_frame, active_fixed_player_ids)

                # Get the stats frame and display it in a separate window
                stats_frame = self.stats_recorder.get_stats_frame()
                cv2.imshow("Game Stats", stats_frame)

                cv2.imshow("Basketball Analysis", annotated_frame)

                self._draw_birds_eye_view(active_fixed_player_ids)

                if cv2.waitKey(1) & 0xFF == ord("q"): break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Processing finished.")

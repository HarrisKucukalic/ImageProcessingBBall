from ultralytics import YOLO
from StatsRecorder import *
from Team import *

# Try to import yt_dlp, if error, set to None
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Centralised definitions for class IDs
PLAYER_CLASS_ID = 3
BALL_CLASS_ID = 0


#  Helper function to find centre of box
def get_bbox_centre(bbox):
    """Calculates the centre of a bounding box."""
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


class BasketballAnalyser:
    """
    Main class to orchestrate the detection, tracking, and team-based analysis.
    """
    def __init__(self, model_path, video_source, tracker_config='botsort.yaml', conf_thresh=0.5, iou_thresh=0.7, start_time="0:00"):
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
    def _initialise_teams(self):
        """
        Initialises the two teams as 'Light' and 'Dark'.
        """
        print("Initialising teams as Light vs. Dark...")
        # BGR format for OpenCV
        team_a_colour = (0, 255, 0)  # Green for the 'Light' team
        team_b_colour = (0, 0, 255)  # Red Grey for the 'Dark' team
        self.stats_recorder.teams['A'] = Team('A', team_a_colour)
        self.stats_recorder.teams['B'] = Team('B', team_b_colour)
        self.teams_initialised = True
        print(f"Teams initialised. Team A (Light): White, Team B (Dark): Grey")

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
            return 'A'  # Team A is 'Light'
        else:
            return 'B'  # Team B is 'Dark'

    def _track_ball(self, detections):
        """
        Refines ball tracking by removing outliers and interpolating short gaps.
        This logic is adapted from the BallTracker class you provided.
        """
        # Find the highest confidence ball detection in the current frame
        ball_detections = [d for d in detections if int(d[6]) == BALL_CLASS_ID]
        if not ball_detections:
            current_ball_centre = None
        else:
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
            return predicted_pos

        # Update History: If we have a valid position, update the history
        if current_ball_centre:
            self.ball_position_history.append(current_ball_centre)
            return current_ball_centre

        return None  # Return None if no ball can be tracked

    def _setup_birds_eye_view(self, frame_shape):
        """Creates the court template with a detailed outline and calculates the homography matrix."""
        h, w, _ = frame_shape
        # Define the dimensions and colours for the top-down court view
        court_w, court_h = 470, 800
        court_colour = (58, 112, 62)  # Green
        line_colour = (255, 255, 255)  # White
        self.court_template = np.zeros((court_h, court_w, 3), dtype=np.uint8)
        self.court_template[:] = court_colour
        # Center circle
        cv2.circle(self.court_template, (court_w // 2, court_h // 2), 50, line_colour, 2)
        # Half-court line
        cv2.line(self.court_template, (0, court_h // 2), (court_w, court_h // 2), line_colour, 2)
        # Top key
        key_width, key_height = 160, 190
        cv2.rectangle(self.court_template, ((court_w - key_width) // 2, 0), ((court_w + key_width) // 2, key_height),line_colour, 2)
        cv2.circle(self.court_template, (court_w // 2, key_height), 60, line_colour, 2)
        # Bottom key
        cv2.rectangle(self.court_template, ((court_w - key_width) // 2, court_h - key_height), ((court_w + key_width) // 2, court_h), line_colour, 2)
        cv2.circle(self.court_template, (court_w // 2, court_h - key_height), 60, line_colour, 2)
        # Three-point arcs
        cv2.ellipse(self.court_template, (court_w // 2, 50), (220, 220), 0, 0, 180, line_colour, 2)
        cv2.ellipse(self.court_template, (court_w // 2, court_h - 50), (220, 220), 0, 180, 360, line_colour, 2)

        # These are estimated points from a video frame (source)
        # and their corresponding locations on the top-down map (destination).
        src_points = np.array([[w * 0.3, h * 0.4], [w * 0.7, h * 0.4], [w * 0.9, h * 0.9], [w * 0.1, h * 0.9]], dtype=np.float32)
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

    def _draw_tracks(self, frame, current_player_ids):
        """Draws dots and labels for players, colour-coded by team."""
        if not self.teams_initialised: return frame
        for player_id, stats in self.stats_recorder.player_stats.items():
            if player_id in current_player_ids and stats.team_id and stats.positions:
                team = self.stats_recorder.teams[stats.team_id]
                dot_colour = team.primary_colour
                centre_x, centre_y = stats.positions[-1]
                cv2.circle(frame, (centre_x, centre_y), 7, dot_colour, -1)

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
        video_url = self.video_source
        video_fps = 0

        if 'youtube.com' in self.video_source or 'youtu.be' in self.video_source:
            try:
                ydl_opts = {'format': 'best[ext=mp4][height<=1080]', 'noplaylist': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.video_source, download=False)
                    video_url, video_fps = info['url'], info.get('fps', 0)
            except Exception as e:
                return print(f"Error extracting YouTube URL: {e}")

        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened(): return print("Error: Could not open video source.")

        if video_fps == 0: video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0: video_fps = 30

        self.stats_recorder = StatsRecorder(video_fps)
        cv2.namedWindow("Basketball Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Basketball Analysis", 1280, 720)
        cv2.namedWindow("Tactical View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tactical View", 470, 800)

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

                if not self.teams_initialised:
                    self._initialise_teams()
                if self.homography_matrix is None:
                    self._setup_birds_eye_view(frame.shape)

                results = \
                self.model.track(source=frame, conf=self.conf_thresh, iou=self.iou_thresh, tracker=self.tracker_config,
                                 persist=True, verbose=False)[0]

                current_player_ids = set()
                if results.boxes.id is not None:
                    detections = results.boxes.data.cpu().numpy()
                    current_player_ids = {int(d[4]) for d in detections if int(d[6]) == PLAYER_CLASS_ID}

                    self.stats_recorder.ball_position = self._track_ball(detections)
                    self.stats_recorder.update(detections, self.frame_number)
                    self._calculate_and_update_gravity(current_player_ids)

                    new_detections = [d for d in detections if int(d[6]) == PLAYER_CLASS_ID and int(
                        d[4]) not in self.stats_recorder.player_stats]

                    for new_player in new_detections:
                        assigned_team = self._get_player_team_assignment(frame, new_player[0:4])
                        if assigned_team:
                            self.stats_recorder.add_player(int(new_player[4]), assigned_team)

                annotated_frame = frame.copy()
                annotated_frame = self._draw_tracks(annotated_frame, current_player_ids)
                annotated_frame = self.stats_recorder.draw_stats(annotated_frame)
                cv2.imshow("Basketball Analysis", annotated_frame)

                self._draw_birds_eye_view(current_player_ids)

                if cv2.waitKey(1) & 0xFF == ord("q"): break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Processing finished.")


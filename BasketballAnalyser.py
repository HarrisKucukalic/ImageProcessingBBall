import cv2
import time
import numpy as np
from ultralytics import YOLO
from StatsRecorder import *
from Team import *

# Try to import yt_dlp
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Centralised definitions for class IDs
PLAYER_CLASS_ID = 3
BALL_CLASS_ID = 0


# --- HELPER FUNCTION ---
def get_bbox_centre(bbox):
    """Calculates the centre of a bounding box."""
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


class BasketballAnalyser:
    """
    Main class to orchestrate the detection, tracking, and team-based analysis process.
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

        # --- CROP TO TORSO REGION ---
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
        # --- END OF CROPPING LOGIC ---

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
        # 1. Find the highest confidence ball detection in the current frame
        ball_detections = [d for d in detections if int(d[6]) == BALL_CLASS_ID]
        if not ball_detections:
            current_ball_centre = None
        else:
            best_ball = max(ball_detections, key=lambda x: x[5])  # Index 5 is confidence
            current_ball_centre = get_bbox_centre(best_ball[0:4])

        # 2. Outlier Rejection: Check if the new position is plausible
        if current_ball_centre and self.ball_position_history:
            last_known_pos = self.ball_position_history[-1]
            distance = np.linalg.norm(np.array(current_ball_centre) - np.array(last_known_pos))

            # If the ball has moved an impossibly large distance, treat it as a misdetection
            if distance > self.max_ball_movement:
                current_ball_centre = None  # Discard the outlier

        # 3. Interpolation: If no valid ball is found, predict its position
        if not current_ball_centre and len(self.ball_position_history) >= 2:
            # Simple linear extrapolation
            last_pos = self.ball_position_history[-1]
            prev_pos = self.ball_position_history[-2]
            velocity = (np.array(last_pos) - np.array(prev_pos))
            # Predict the next position based on the last known velocity
            predicted_pos = tuple(map(int, np.array(last_pos) + velocity))
            return predicted_pos

        # 4. Update History: If we have a valid position, update the history
        if current_ball_centre:
            self.ball_position_history.append(current_ball_centre)
            return current_ball_centre

        return None  # Return None if no ball can be tracked

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

                results = \
                self.model.track(source=frame, conf=self.conf_thresh, iou=self.iou_thresh, tracker=self.tracker_config,
                                 persist=True, verbose=False)[0]

                if results.boxes.id is not None:
                    detections = results.boxes.data.cpu().numpy()
                    current_player_ids = {int(d[4]) for d in detections if int(d[6]) == PLAYER_CLASS_ID}
                    # 1. Get the refined ball position
                    refined_ball_pos = self._track_ball(detections)

                    # 2. Update the StatsRecorder with all detections, then overwrite the ball position
                    self.stats_recorder.update(detections)
                    self.stats_recorder.ball_position = refined_ball_pos

                    new_detections = [d for d in detections if int(d[6]) == PLAYER_CLASS_ID and int(
                        d[4]) not in self.stats_recorder.player_stats]

                    for new_player_detection in new_detections:
                        new_player_id = int(new_player_detection[4])
                        player_bbox = new_player_detection[0:4]

                        assigned_team = self._get_player_team_assignment(frame, player_bbox)
                        if assigned_team:
                            self.stats_recorder.add_player(new_player_id, assigned_team)

                annotated_frame = frame.copy()
                annotated_frame = self._draw_tracks(annotated_frame, current_player_ids)
                annotated_frame = self.stats_recorder.draw_stats(annotated_frame)
                cv2.imshow("Basketball Analysis", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"): break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Processing finished.")


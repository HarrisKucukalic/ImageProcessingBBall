from PlayerStats import *
import numpy as np
import cv2
CLASS_NAMES = {
    0: 'Ball',
    1: 'Hoop',
    2: 'Period',
    3: 'Player',
    4: 'Ref',
    5: 'Shot Clock',
    6: 'Team Name',
    7: 'Team Points',
    8: 'Time Remaining'
}
PLAYER_CLASS_ID = 3
BALL_CLASS_ID = 0
TRAIL_LENGTH = 25
BALL_POSSESSION_THRESHOLD = 50
class StatsRecorder:
    """Manages all player statistics and determines game-level events like possession."""

    def __init__(self, fps):
        # Maps track_id to PlayerStats object
        self.player_stats = {}
        self.ball_position = None
        self.player_with_ball_id = None
        self.time_per_frame = 1.0 / fps if fps > 0 else 0
        self.ball_positions = deque(maxlen=TRAIL_LENGTH)

    def _update_ball_possession(self):
        """
        Determines which player is closest to the ball and grants possession
        if they are within the defined threshold.
        """
        self.player_with_ball_id = None
        if self.ball_position is None or not self.players:
            return

        min_dist = float('inf')
        closest_player_id = None

        for track_id, player in self.players.items():
            if player.positions:
                # Get the latest position
                player_pos = player.positions[-1]
                dist = np.linalg.norm(np.array(player_pos) - np.array(self.ball_position))
                if dist < min_dist:
                    min_dist = dist
                    closest_player_id = track_id

        if closest_player_id is not None and min_dist < BALL_POSSESSION_THRESHOLD:
            self.player_with_ball_id = closest_player_id

    def update(self, tracks):
        """Updates stats based on the latest tracking data."""
        current_player_positions = {}
        self.ball_position = None
        self.player_with_ball_id = None  # Reset each frame

        # First, find the ball and all player positions
        for box, track_id, cls_id in tracks:
            if cls_id == BALL_CLASS_ID:
                # Get the center of the ball
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                self.ball_position = (x_center, y_center)
                self.ball_positions.append(self.ball_position)
            elif cls_id == PLAYER_CLASS_ID:
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                current_player_positions[track_id] = (x_center, y_center)
                if track_id not in self.player_stats:
                    self.player_stats[track_id] = PlayerStats(track_id)
                self.player_stats[track_id].positions.append((int(x_center), int(y_center)))

        # If a ball is detected, find the closest player
        if self.ball_position:
            min_dist = float('inf')
            closest_player_id = None
            for track_id, player_pos in current_player_positions.items():
                dist = np.linalg.norm(np.array(self.ball_position) - np.array(player_pos))
                if dist < min_dist and dist < BALL_POSSESSION_THRESHOLD:
                    min_dist = dist
                    closest_player_id = track_id

            if closest_player_id is not None:
                self.player_with_ball_id = closest_player_id
                self.player_stats[closest_player_id].possession_time += self.time_per_frame

    def draw_stats(self, frame):
        """Draws player trails and a stats overlay on the frame."""
        if len(self.ball_positions) > 1:
            pts = np.array(self.ball_positions, dtype=np.int32).reshape((-1, 1, 2))
            # Draw the ball trail in a distinct color (e.g., red) and thicker
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=3)

        # Draw a stats box
        box_x, box_y, box_w, box_h = 10, 10, 300, 150
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 1, cv2.LINE_AA)

        # Display possession info
        text_y = box_y + 25
        title_text = "Possession Stats"
        cv2.putText(frame, title_text, (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        text_y += 30
        if self.player_with_ball_id is not None:
            poss_time = self.player_stats[self.player_with_ball_id].possession_time
            poss_text = f"Player {self.player_with_ball_id}: {poss_time:.1f}s"
            cv2.putText(frame, poss_text, (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No possession", (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return frame
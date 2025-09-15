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
HOOP_CLASS_ID = 1
TRAIL_LENGTH = 25
BALL_POSSESSION_THRESHOLD = 50
class StatsRecorder:
    """Manages all player statistics and determines game-level events like possession."""

    def __init__(self, fps):
        # Maps track_id to PlayerStats object
        self.player_stats = {}
        self.ball_position = None
        self.hoop_position = None
        self.player_with_ball_id = None
        self.time_per_frame = 1.0 / fps if fps > 0 else 0
        self.ball_positions = deque(maxlen=TRAIL_LENGTH)
        self.ball_above_hoop = False

    def _check_for_score(self):
        """Checks if the ball has passed through the hoop and assigns score to the player with possession."""
        if self.ball_position is None or self.hoop_position is None:
            return

        ball_cx, ball_cy = self.ball_position
        hoop_x1, hoop_y1, hoop_x2, hoop_y2 = self.hoop_position
        hoop_cy = (hoop_y1 + hoop_y2) / 2

        # Check if the ball's center is horizontally within the hoop's bounding box
        is_overlapping = hoop_x1 < ball_cx < hoop_x2

        if is_overlapping:
            # If the ball is above the hoop's center, set the flag
            if ball_cy < hoop_cy:
                self.ball_above_hoop = True
            # If the ball is now below the hoop and the 'above' flag was set, it's a score
            elif ball_cy >= hoop_cy and self.ball_above_hoop:
                # If a player has possession, attribute the score to them
                if self.player_with_ball_id in self.player_stats:
                    self.player_stats[self.player_with_ball_id].score += 1
                    print(f"Score detected for Player {self.player_with_ball_id}!")
                else:
                    print("Score detected, but no player had possession.")

                # Reset the flag to prevent counting the same shot multiple times
                self.ball_above_hoop = False
        else:
            # Reset the flag if the ball is no longer overlapping with the hoop
            self.ball_above_hoop = False

    def update(self, tracks):
        """Updates stats based on the latest tracking data."""
        current_player_positions = {}
        self.ball_position = None
        self.hoop_position = None  # Reset hoop position each frame
        self.player_with_ball_id = None  # Reset each frame

        # First, find the ball, hoop, and all player positions
        for box, track_id, cls_id in tracks:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2

            if cls_id == BALL_CLASS_ID:
                self.ball_position = (x_center, y_center)
                self.ball_positions.append(self.ball_position)
            elif cls_id == HOOP_CLASS_ID:
                self.hoop_position = box  # Store the entire bounding box
            elif cls_id == PLAYER_CLASS_ID:
                current_player_positions[track_id] = (x_center, y_center)
                if track_id not in self.player_stats:
                    self.player_stats[track_id] = PlayerStats(track_id)
                self.player_stats[track_id].positions.append((int(x_center), int(y_center)))

        # If a ball is detected, find the closest player for possession
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

        # Check for a score in every frame
        self._check_for_score()

    def draw_stats(self, frame):
        """Draws player trails and a stats overlay on the frame."""

        # Dynamically adjust box height based on the number of players
        num_players = len(self.player_stats)
        box_h = 70 + (num_players * 25)  # Base height + 25px per player
        box_x, box_y, box_w = 10, 10, 300

        # Draw a semi-transparent box for the stats
        sub_img = frame[box_y:box_y + box_h, box_x:box_x + box_w]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.6, black_rect, 0.4, 1.0)
        frame[box_y:box_y + box_h, box_x:box_x + box_w] = res

        # Draw border
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 1, cv2.LINE_AA)

        # Display Title
        text_y = box_y + 25
        cv2.putText(frame, "Player Scoreboard", (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display each player's score
        for player_id, stats in self.player_stats.items():
            text_y += 25
            score_text = f"Player {player_id}: {stats.score}"
            # Highlight the player with possession
            color = (0, 255, 0) if player_id == self.player_with_ball_id else (255, 255, 255)
            cv2.putText(frame, score_text, (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame
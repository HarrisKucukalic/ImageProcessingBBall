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
BALL_POSSESSION_THRESHOLD = 50
class StatsRecorder:
    """Manages all player statistics and determines game-level events like possession."""

    def __init__(self):
        # Maps track_id to PlayerStats object
        self.players = {}
        self.ball_position = None
        self.player_with_ball_id = None

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
        """
        Main update loop. Processes tracking data to update player positions and ball possession.
        'tracks' is a numpy array from BoT-SORT: [x1, y1, x2, y2, track_id, class_id, conf]
        """
        current_track_ids = set()
        self.ball_position = None

        # First pass: update positions and find the ball
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, _ = track
            track_id = int(track_id)
            class_id = int(class_id)
            class_name = CLASS_NAMES.get(class_id, 'unknown')

            bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            if class_name == 'basketball':
                self.ball_position = bbox_center
            elif class_name == 'player':
                current_track_ids.add(track_id)
                if track_id not in self.players:
                    self.players[track_id] = PlayerStats(track_id)
                self.players[track_id].update_position(bbox_center)

        # Remove players that are no longer being tracked
        lost_track_ids = set(self.players.keys()) - current_track_ids
        for track_id in lost_track_ids:
            del self.players[track_id]

        # Second pass: determine possession and update timers
        self._update_ball_possession()
        for track_id, player in self.players.items():
            has_ball = (track_id == self.player_with_ball_id)
            player.update_possession(has_ball)

    def draw_stats_overlay(self, frame):
        """Draws the statistics panel on the frame."""
        overlay = frame.copy()
        alpha = 0.6
        cv2.rectangle(overlay, (10, 10), (250, 20 + len(self.players) * 20), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        y_pos = 30
        cv2.putText(frame, "Player Stats:", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25

        sorted_players = sorted(self.players.items())

        for track_id, player in sorted_players:
            text = f"P{track_id}: {player.possession_time:.1f}s"
            color = (0, 255, 255) if player.has_ball else (255, 255, 255)
            cv2.putText(frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_pos += 20
        return frame
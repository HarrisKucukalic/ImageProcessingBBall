from collections import deque
import time
class PlayerStats:
    """A simple class to store and manage statistics for a single tracked player."""

    def __init__(self, track_id):
        self.track_id = track_id
        self.has_ball = False
        self.possession_time = 0.0
        self.last_update_time = None
        # Use a double-ended queue to store recent positions for drawing a trail
        self.positions = deque(maxlen=30)
        self.score = 0
        self.last_seen_frame = 0
        self.last_bbox = None

    def update_position(self, bbox_center):
        """Adds the latest bounding box center to the player's position history."""
        self.positions.append(bbox_center)

    def update_possession(self, has_ball_now: bool):
        """Updates the player's ball possession status and timer."""
        current_time = time.time()
        if self.last_update_time is None:
            self.last_update_time = current_time

        if has_ball_now:
            if not self.has_ball:
                # Player just gained possession
                self.has_ball = True
                self.last_update_time = current_time
            else:
                # Player continues to have possession, add elapsed time
                self.possession_time += (current_time - self.last_update_time)
                self.last_update_time = current_time
        else:
            # Player does not have the ball
            self.has_ball = False

        # Always update last_update_time to prevent stale calculations
        if not self.has_ball:
            self.last_update_time = current_time
from collections import deque
import numpy as np
import cv2
from PlayerStats import *
from Team import *

# Class IDs
PLAYER_CLASS_ID = 3
BALL_CLASS_ID = 0
HOOP_CLASS_ID = 1
BALL_POSSESSION_THRESHOLD = 75

class StatsRecorder:
    """Manages all team and player statistics."""

    def __init__(self, fps):
        self.player_stats = {}
        self.teams = {'A': None, 'B': None}
        self.ball_position = None
        self.hoop_position = None
        self.player_with_ball_id = None
        self.team_with_ball_id = None  # Tracks which team has the ball
        self.last_player_with_ball = None  # Remembers the last player with possession for scoring
        self.time_per_frame = 1.0 / fps if fps > 0 else 0
        self.ball_above_hoop = False

    def add_player(self, player_id, team_id):
        """Adds a new player to the stats tracker and their respective team."""
        if player_id not in self.player_stats:
            self.player_stats[player_id] = PlayerStats(player_id, team_id)
            if team_id in self.teams and self.teams[team_id]:
                self.teams[team_id].add_player(player_id)

    def _check_for_score(self):
        """Checks if a score has occurred and attributes it to the correct team."""
        if self.ball_position is None or self.hoop_position is None:
            return

        ball_cx, ball_cy = self.ball_position
        hoop_x1, hoop_y1, hoop_x2, hoop_y2 = self.hoop_position
        hoop_cy = (hoop_y1 + hoop_y2) / 2

        is_overlapping = hoop_x1 < ball_cx < hoop_x2
        if is_overlapping:
            if ball_cy < hoop_cy:
                self.ball_above_hoop = True
            elif ball_cy >= hoop_cy and self.ball_above_hoop:
                scorer = self.player_stats.get(self.last_player_with_ball)
                if scorer and scorer.team_id in self.teams:
                    self.teams[scorer.team_id].score += 2
                    print(f"Score for Team {scorer.team_id}!")
                self.ball_above_hoop = False
        else:
            self.ball_above_hoop = False

    def update(self, detections):
        """Updates all stats based on the latest frame's detections."""
        current_player_positions = {}
        self.hoop_position = None
        self.player_with_ball_id = None  # Reset each frame

        for d in detections:
            box, track_id, cls_id = d[0:4], int(d[4]), int(d[6])

            # Helper function to get the centre of a bounding box
            def get_bbox_centre(bbox):
                return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

            centre = get_bbox_centre(box)

            if cls_id == PLAYER_CLASS_ID:
                if track_id in self.player_stats:
                    self.player_stats[track_id].positions.append(centre)
                current_player_positions[track_id] = centre
            elif cls_id == HOOP_CLASS_ID:
                self.hoop_position = box

        # Determine player and team possession based on the refined ball position
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
                self.last_player_with_ball = closest_player_id

                # Update team possession
                player_stats = self.player_stats.get(closest_player_id)
                if player_stats and player_stats.team_id:
                    if self.team_with_ball_id != player_stats.team_id:
                        print(f"Possession changed to Team {player_stats.team_id}")
                    self.team_with_ball_id = player_stats.team_id

        self._check_for_score()

    def draw_stats(self, frame):
        """Draws the main scoreboard, including team possession."""
        if not self.teams.get('A') or not self.teams.get('B'):
            return frame

        team_a = self.teams['A']
        team_b = self.teams['B']

        box_x, box_y, box_w, box_h = 10, 10, 250, 120

        sub_img = frame[box_y:box_y + box_h, box_x:box_x + box_w]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.6, black_rect, 0.4, 1.0)
        frame[box_y:box_y + box_h, box_x:box_x + box_w] = res
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 1)

        # Title
        text_y = box_y + 25
        cv2.putText(frame, "SCOREBOARD", (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Team A (Light)
        text_y += 35
        team_a_text = f"Light Team: {team_a.score}"
        team_a_colour = (0, 0, 0)  # Black text
        # Highlight with possession indicator
        if self.team_with_ball_id == 'A':
            team_a_text += " (P)"
            team_a_colour = (0, 255, 0)  # Green text to show possession
        cv2.putText(frame, team_a_text, (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, team_a_colour, 2)

        # Team B (Dark)
        text_y += 30
        team_b_text = f"Dark Team: {team_b.score}"
        team_b_colour = (255, 255, 255)  # White text
        if self.team_with_ball_id == 'B':
            team_b_text += " (P)"
            team_b_colour = (0, 255, 0)
        cv2.putText(frame, team_b_text, (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, team_b_colour, 2)

        return frame

    def draw_stats(self, frame):
        """Draws the team scoreboard on the frame."""
        if self.teams['A'] is None: return frame  # Don't draw if teams aren't ready

        box_x, box_y, box_w, box_h = 10, 10, 300, 120
        sub_img = frame[box_y:box_y + box_h, box_x:box_x + box_w]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.6, black_rect, 0.4, 1.0)
        frame[box_y:box_y + box_h, box_x:box_x + box_w] = res
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 1)

        text_y = box_y + 25
        cv2.putText(frame, "Team Scores", (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Team A Score
        text_y += 35
        team_a = self.teams['A']
        colour_a = tuple(c for c in team_a.primary_colour)
        cv2.putText(frame, f"Team A: {team_a.score}", (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour_a, 2)

        # Team B Score
        text_y += 35
        team_b = self.teams['B']
        colour_b = tuple(c for c in team_b.primary_colour)
        cv2.putText(frame, f"Team B: {team_b.score}", (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour_b, 2)

        return frame


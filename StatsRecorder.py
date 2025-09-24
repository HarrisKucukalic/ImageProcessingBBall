import cv2
import numpy as np
from PlayerStats import PlayerStats
from Team import Team
from Team import *
HOOP_CLASS_ID = 1

class StatsRecorder:
    """
    Manages all game statistics, including scores, possession, and player data.
    """
    def __init__(self, video_fps, team_a, team_b):
        self.video_fps = video_fps
        self.player_stats = {}
        self.teams = {team_a.team_id: team_a, team_b.team_id: team_b}
        self.ball_position = None
        self.hoop_position = None
        self.last_score_frame = -100  # Cooldown to prevent duplicate scores
        self.player_with_ball = None
        self.possession_team = None
        self.gravity_score = 0.0
        self.highest_gravity_player_id = None

    def add_player(self, player_id, team_id):
        """Adds a new player to the stats' tracker."""
        if player_id not in self.player_stats:
            new_player = PlayerStats(player_id, team_id)
            self.player_stats[player_id] = new_player
            if team_id in self.teams and self.teams[team_id]:
                self.teams[team_id].add_player(new_player)
            print(f"Added Player {player_id} to Team {team_id}")

    def update(self, detections, frame_number):
        """Updates player positions, checks for scores, and determines possession."""
        # Update hoop position first if detected
        hoop_detections = [d for d in detections if int(d[6]) == HOOP_CLASS_ID]
        if hoop_detections:
            # x[5] = confidence score, this find the highest confidence score and then extracts the first 4 elements, which are the four corners of the box
            self.hoop_position = max(hoop_detections, key=lambda x: x[5])[0:4]

        # Update player positions
        for d in detections:
            player_id = int(d[4])
            if player_id in self.player_stats:
                centre_x = int((d[0] + d[2]) / 2)
                centre_y = int((d[1] + d[3]) / 2)
                self.player_stats[player_id].update_position((centre_x, centre_y))

        self._check_for_score(frame_number)
        self._update_possession()

    def _check_for_score(self, frame_number):
        """Checks if the ball is inside the hoop, and if so, attributes a score."""
        if frame_number < (self.last_score_frame + self.video_fps * 2):
            return

        if self.ball_position is not None and self.hoop_position is not None:
            ball_x, ball_y = self.ball_position
            h_x1, h_y1, h_x2, h_y2 = self.hoop_position

            if h_x1 < ball_x < h_x2 and h_y1 < ball_y < h_y2:
                if self.possession_team and self.possession_team in self.teams:
                    # Update player points first
                    player_with_ball_stats = self.player_stats.get(self.player_with_ball)
                    if player_with_ball_stats:
                        player_with_ball_stats.update_points(2)

                    # Update the team score using the Team's method
                    team_with_possession = self.teams.get(self.possession_team)
                    if team_with_possession:
                        team_with_possession.update_score()

                    print(f"Score for Team {self.possession_team}!")
                    self.last_score_frame = frame_number

    def _update_possession(self):
        """Determines which player and team has possession of the ball."""
        if not self.ball_position:
            self.player_with_ball = None
            self.possession_team = None
            return

        min_dist = float('inf')
        player_with_ball_id = None

        for player_id, stats in self.player_stats.items():
            if stats.positions:
                last_pos = np.array(stats.positions[-1])
                dist = np.linalg.norm(last_pos - np.array(self.ball_position))

                if dist < min_dist and dist < 35:
                    min_dist = dist
                    player_with_ball_id = player_id

        if player_with_ball_id:
            if self.player_with_ball != player_with_ball_id:
                self.player_with_ball = player_with_ball_id
                new_possession_team = self.player_stats[player_with_ball_id].team_id
                if self.possession_team != new_possession_team:
                    self.possession_team = new_possession_team
                    print(f"Possession changed to Team {self.possession_team}")
        else:
            self.player_with_ball = None
            self.possession_team = None

    def get_stats_frame(self):
        """Creates and returns an image frame of the game statistics."""
        # Create a blank frame for the stats display
        frame_width, frame_height = 400, 250
        stats_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Draw a background rectangle for the scoreboard
        cv2.rectangle(stats_frame, (10, 10), (frame_width - 10, frame_height - 10), (0, 0, 0), -1)

        if not self.teams.get('A') or not self.teams.get('B'):
            return stats_frame

        team_a_key = next(iter(self.teams))
        team_b_key = next(iter(x for x in self.teams if x != team_a_key), None)

        if not team_b_key:
            return stats_frame

        team_a = self.teams[team_a_key]
        team_b = self.teams[team_b_key]

        possession_a = " (P)" if self.possession_team == team_a.team_id else ""
        possession_b = " (P)" if self.possession_team == team_b.team_id else ""

        score_text_a = f"Team {team_a.team_id}: {team_a.score}{possession_a}"
        score_text_b = f"Team {team_b.team_id}: {team_b.score}{possession_b}"

        cv2.putText(stats_frame, score_text_a, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, team_a.primary_colour, 2)
        cv2.putText(stats_frame, score_text_b, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, team_b.primary_colour, 2)

        gravity_text = f"Defensive Gravity: {self.gravity_score:.2f}"
        cv2.putText(stats_frame, gravity_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Display individual player stats
        y_offset = 150
        for player_id, stats in self.player_stats.items():
            player_info_text = f"P{player_id} ({stats.team_id}): {stats.points} pts"
            cv2.putText(stats_frame, player_info_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        1)
            y_offset += 25

        return stats_frame
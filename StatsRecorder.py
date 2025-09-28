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
        self.current_frame_number = 0  # To track game time

        # --- Shot Detection Stats ---
        self.makes = 0
        self.attempts = 0

    @property
    def current_time_string(self):
        """Calculates the current game time in MM:SS format based on frame count and FPS."""
        total_seconds = int(self.current_frame_number / self.video_fps)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def add_player(self, player_id, team_id):
        """Adds a new player to the stats' tracker."""
        if player_id not in self.player_stats:
            new_player = PlayerStats(player_id, team_id)
            self.player_stats[player_id] = new_player
            if team_id in self.teams and self.teams[team_id]:
                self.teams[team_id].add_player(new_player)
            print(f"Added Player {player_id} to Team {team_id}")

    def remove_player(self, player_id):
        """Removes a player from the stats tracker."""
        if player_id in self.player_stats:
            team_id = self.player_stats[player_id].team_id
            if team_id in self.teams and self.teams[team_id]:
                self.teams[team_id].remove_player(player_id)
            del self.player_stats[player_id]
            print(f"Removed Player {player_id} from Team {team_id}")
        else:
            print(f"Player {player_id} not found in stats.")

    def update(self, detections, frame_number):
        """Updates player positions and determines possession."""
        self.current_frame_number = frame_number

        # Update hoop position if detected (needed for hoop tracking/gravity, but scoring is handled by shot logic)
        hoop_detections = [d for d in detections if int(d[6]) == HOOP_CLASS_ID]
        if hoop_detections:
            self.hoop_position = max(hoop_detections, key=lambda x: x[5])[0:4]

        self._update_possession()

        # Note: Makes/attempts are updated by the Shot Detection logic in BasketballAnalyser

    # Removed the old _check_for_score(frame_number) method as shot logic now handles scoring and points updates.

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
                # We expect ball to be closer than 35 pixels to the player for possession
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

        # --- Line 1: Game Time ---
        time_text = f"Time: {self.current_time_string}"
        cv2.putText(stats_frame, time_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Line 2: Team Scores ---
        score_text_a = f"Team {team_a.team_id}: {team_a.score}{possession_a}"
        score_text_b = f"Team {team_b.team_id}: {team_b.score}{possession_b}"

        cv2.putText(stats_frame, score_text_a, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, team_a.primary_colour, 2)
        cv2.putText(stats_frame, score_text_b, (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, team_b.primary_colour, 2)

        # --- Line 3: Shot Stats ---
        shot_text = f"Shots: {self.makes} / {self.attempts}"
        cv2.putText(stats_frame, shot_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        # --- Line 4: Defensive Gravity ---
        gravity_text = f"Defensive Gravity: {self.gravity_score:.2f}"
        cv2.putText(stats_frame, gravity_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Display individual player stats (Space is limited, showing fewer lines)
        y_offset = 150
        # for player_id, stats in self.player_stats.items():
        #     player_info_text = f"P{player_id} ({stats.team_id}): {stats.points} pts"
        #     cv2.putText(stats_frame, player_info_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
        #                 1)
        #     y_offset += 25

        return stats_frame

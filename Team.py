from PlayerStats import *


class Team:
    """Represents a single team, tracking its colour, score, and player roster."""

    def __init__(self, team_id, primary_colour):
        self.team_id = team_id
        self.primary_colour = primary_colour
        self.score = 0
        # Use a dictionary for the roster for efficient adding and counting
        self.roster = {}

    def add_player(self, player_stats: PlayerStats):
        """Adds a PlayerStats object to the team's roster."""
        # Use the player's track_id as the key
        self.roster[player_stats.track_id] = player_stats

    def remove_player(self, player_id: int):
        """Removes a player's PlayerStats object from the team's roster."""
        if player_id in self.roster:
            del self.roster[player_id]

    def get_player(self, player_id: int):
        """Returns the PlayerStats object for a given player_id."""
        return self.roster.get(player_id)

    def get_player_count(self):
        """Returns the current number of players on the team's roster."""
        return len(self.roster)

    def get_all_player_stats(self):
        """Returns a list of all PlayerStats objects on the team."""
        return list(self.roster.values())

    def update_score(self):
        total_player_points = 0
        for players in self.roster.values():
            total_player_points += players.points
        self.score = total_player_points

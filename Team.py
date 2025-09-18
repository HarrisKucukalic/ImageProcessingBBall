class Team:
    """Represents a single team, tracking its colour, score, and player roster."""

    def __init__(self, team_id, primary_colour):
        self.team_id = team_id
        self.primary_colour = primary_colour
        self.score = 0
        # Use a set for the roster for efficient adding and counting
        self.roster = set()

    def add_player(self, player_id):
        """Adds a player's ID to the team's roster."""
        self.roster.add(player_id)

    def remove_player(self, player_id):
        """Removes a player's ID from the team's roster."""
        self.roster.discard(player_id) # Use discard to avoid errors if player isn't found

    def get_player_count(self):
        """Returns the current number of players on the team's roster."""
        return len(self.roster)
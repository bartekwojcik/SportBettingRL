from datetime import datetime
from betting_env.enums.bet_enum import WinnerEnum


class Event:
    """
    Single record representation
    """
    def __init__(
        self,
        event_id: int,
        event_date: datetime,
        away_team: int,
        home_team: int,
        league: int,
        away_score: int,
        home_score: int,
        away_odds: float,
        home_odds: float,
        draw_odds: float,
    ):
        self.draw_odds = draw_odds
        self.event_id = event_id
        self.event_date = event_date
        self.home_odds = home_odds
        self.away_odds = away_odds
        self.home_score = home_score
        self.away_score = away_score
        self.league = league
        self.home_team = home_team
        self.away_team = away_team

    def get_winner(self) -> WinnerEnum:

        if self.home_score > self.away_score:
            return WinnerEnum.home_won
        elif self.home_score < self.away_score:
            return WinnerEnum.away_won
        elif self.home_score == self.away_score:
            return WinnerEnum.draw_won
        else:
            raise ValueError("who won?")

    def __repr__(self):
        return f"{self.event_id}, {self.event_date}"

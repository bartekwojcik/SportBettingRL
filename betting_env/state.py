import numpy as np
from betting_env.event import Event
from sklearn.preprocessing import normalize


class EnvState:
    """
    Representation of state in environment
    """

    def __init__(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        event_day: int,
        event_month: int,
        event_year: int,
        home_odds: float,
        away_odds: float,
        draw_odds: float,
        current_bankroll: float,
        original_event: Event,
    ):

        self._original_event = original_event
        self.current_bankroll = current_bankroll
        self.draw_odds = draw_odds
        self.away_odds = away_odds
        self.home_odds = home_odds
        self.event_year = event_year
        self.event_month = event_month
        self.event_day = event_day
        self.league_id = league_id
        self.away_team_id = away_team_id
        self.home_team_id = home_team_id

    def to_vector(self) -> np.ndarray:

        vector = np.array(
            [
                self.current_bankroll,
                self.draw_odds,
                self.away_odds,
                self.home_odds,
                self.event_year,
                self.event_month,
                self.event_day,
                self.league_id,
                self.away_team_id,
                self.home_team_id,
            ]
        )

        return vector

    def to_normalized_vector(self, initial_bankroll) -> np.ndarray:
        """
        Returns shorter version of vector with normalized odds and bankroll

        :param initial_bankroll:
        :return:
        """

        norm_odds = normalize(
            np.array([self.draw_odds, self.away_odds, self.home_odds]).reshape(1, -1)
        ).reshape(-1,)
        normalized_bankroll = (
            np.array([self.current_bankroll / initial_bankroll])
            if initial_bankroll != 0
            else self.current_bankroll
        )
        vector = np.concatenate((normalized_bankroll, norm_odds))

        return vector

    def get_original_event(self) -> Event:
        """
        :return: Original event with all it's information
        """
        return self._original_event

    def __repr__(self):
        value = (
            f""
            f"current_bankroll: {self.current_bankroll} "
            f"draw_odds: {self.draw_odds} "
            f"away_odds: {self.away_odds} "
            f"home_odds: {self.home_odds} "
            f"event_year: {self.event_year} "
            f"event_month: {self.event_month} "
            f"event_day: {self.event_day} "
            f"league_id: {self.league_id} "
            f"away_team_id: {self.away_team_id} "
            f"home_team_id: {self.home_team_id} "
        )

        return value

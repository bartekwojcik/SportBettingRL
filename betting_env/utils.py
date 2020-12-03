from sklearn.preprocessing import LabelEncoder
from typing import Dict

from betting_env.event import Event
from betting_env.state import EnvState


def label_encoder_to_dictionary(le: LabelEncoder) -> Dict[str, int]:
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    return le_dict


def convert_event_to_state(event: Event, current_bankroll: float) -> EnvState:

    event_day = event.event_date.day
    event_year = event.event_date.year
    event_month = event.event_date.month

    state = EnvState(
        home_team_id=event.home_team,
        away_team_id=event.away_team,
        league_id=event.league,
        event_day=event_day,
        event_month=event_month,
        event_year=event_year,
        home_odds=event.home_odds,
        away_odds=event.away_odds,
        draw_odds=event.draw_odds,
        current_bankroll=current_bankroll,
        original_event=event,
    )

    return state

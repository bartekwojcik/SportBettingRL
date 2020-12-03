from enum import Enum


class EnumIntComparisonMixin:
    def __eq__(self, value):
        return self.value == value


class WinnerEnum(EnumIntComparisonMixin, Enum):
    home_won = 0
    away_won = 1
    draw_won = 2


class ActionEnum(EnumIntComparisonMixin, Enum):
    do_nothing = 0
    bet_home = 1
    bet_away = 2
    bet_draw = 3

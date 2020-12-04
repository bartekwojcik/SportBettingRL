import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Union, Optional

from betting_env.data_loading_strategies.csv_column_mapping import CsvColumnsMappings
from betting_env.event import Event
import multiprocessing
import dask.dataframe as dd


class EventFactory:
    """
    Manages events from CSV
    """

    def __init__(
        self, input_dataframe: pd.DataFrame, column_mapping: CsvColumnsMappings,
    ):
        """
        :param input_dataframe: pandas dataframe with loaded csv file
        :param column_mapping: mapping of csv's columns
        """
        self._column_mapping = column_mapping
        self._df = input_dataframe.copy()
        self._event_list = None

    def get_all_events(self) -> List[Event]:
        """
        Reads dataframe and converts it to list of events

        :return: List of Events
        """

        if self._event_list == None:
            num_cores = multiprocessing.cpu_count() - 1
            df_dask = dd.from_pandas(self._df, npartitions=num_cores)

            event_df = df_dask.apply(
                lambda row: self._convert_row_to_event(row),
                axis=1,
                meta=(None, "object"),
            ).compute(scheduler="multiprocessing")
            self._event_list = event_df.to_list()

        return self._event_list

    def _convert_row_to_event(self, row):

        cm = self._column_mapping
        event = Event(
            event_id=row[cm.event_id],
            event_date=row[cm.event_date],
            away_team=row[cm.away_team],
            home_team=row[cm.home_team],
            league=row[cm.league],
            away_score=row[cm.away_score],
            home_score=row[cm.home_score],
            away_odds=row[cm.away_odds],
            home_odds=row[cm.home_odds],
            draw_odds=row[cm.draw_odds],
        )

        return event

    def is_event_available(self, index: int) -> bool:
        """
        Checks if event of given index exists

        :param index: index of event
        :return: bool
        """
        all_events = self.get_all_events()
        if len(all_events) <= index:
            return False
        return True

    def get_event_by_index(self, index: int) -> Event:
        """
        :param index: index of event
        :return: Event
        """
        all_events = self.get_all_events()

        if len(all_events) <= index:
            raise ValueError("Index is greater than length of event list")

        event = all_events[index]

        return event

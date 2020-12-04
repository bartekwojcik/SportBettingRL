import pandas as pd
from typing import List
from betting_env.event import Event
from betting_env.data_loading_strategies.csv_column_mapping import CsvColumnsMappings
import multiprocessing
import dask.dataframe as dd


class CSVLoadingStrategy:
    def __init__(self, csv_file_path: str, column_mapping: CsvColumnsMappings):
        """
        Orders events by Event date column (specified in column mapping object)
        :param csv_file_path:
        :param column_mapping:
        """
        self._column_mapping = column_mapping
        self._df = pd.read_csv(csv_file_path, sep=self._column_mapping.column_separator)
        self._event_list = None  # todo this goes to event creator/factory

        self._order_dataframe()

    def _order_dataframe(self) -> None:

        num_cores = multiprocessing.cpu_count() - 1
        ddf = dd.from_pandas(self._df, npartitions=num_cores)

        self._df[self._column_mapping.event_date] = dd.to_datetime(
            self._df[self._column_mapping.event_date],
            format=self._column_mapping.date_format,
        )

        self._df = self._df.sort_values(by=self._column_mapping.event_date)
        self._df = self._df.reset_index(drop=True)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

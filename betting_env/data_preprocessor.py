import pandas as pd
from typing import Dict, Tuple, List
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:

    # todo columns_to_standarize, columns_to_normalize

    def encode_columns(
        self, input_df: pd.DataFrame, columns_to_encode: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """Creates a copy of dataframe and encodes selected columns.

        :param columns_to_encode:
        :return: Tuple of:
        (pandas dataframe with encoded columns,
        Dict of label encoders in the same order as in `columns_to_encode` column_name->Encoder)
        """

        input_df = input_df.copy()
        encoders = {}

        for column_name in columns_to_encode:
            if column_name in input_df.columns:
                lb = LabelEncoder()
                input_df[column_name] = lb.fit_transform(input_df[column_name])
                encoders[column_name] = lb

            else:
                raise ValueError(
                    f"column name '{column_name}' does not exist in dataframe"
                )

        return (input_df, encoders)

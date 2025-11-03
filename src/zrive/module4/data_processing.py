import pandas as pd
import logging
from typing import Optional


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] - %(funcName)s - %(message)s')


def preprocess_data_for_training(df: pd.DataFrame,
                                 min_items: int = 5,
                                 columns_to_drop: Optional[list[str]] = None
                                 ) -> tuple[pd.DataFrame, pd.DataFrame,
                                            pd.DataFrame, pd.DataFrame,
                                            pd.DataFrame, pd.DataFrame]:
    '''
    Preprocesses the DataFrame to get the data ready for modeling.
    '''

    logger.info(f"Filtering orders with at least {min_items} items...")
    df = df.loc[df.groupby('order_id').filter(lambda x: (x.outcome == 1).sum() >= min_items).index
                ].sort_values(by='created_at')

    logger.info("Splitting data into train, validation, and test sets...")
    x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_split(df)

    logger.info("Applying common transformations...")
    x_train, binary_cols, categorical_cols, numerical_cols = apply_common_transformations(x_train)
    x_val, _, _, _ = apply_common_transformations(x_val)
    x_test, _, _, _ = apply_common_transformations(x_test)

    logger.info(f"Dropping categorical columns: {categorical_cols}...")
    x_train.drop(columns=categorical_cols, inplace=True)
    x_val.drop(columns=categorical_cols, inplace=True)
    x_test.drop(columns=categorical_cols, inplace=True)

    if columns_to_drop:
        logger.info(f"Dropping additional columns: {columns_to_drop}...")
        x_train.drop(columns=columns_to_drop, inplace=True)
        x_val.drop(columns=columns_to_drop, inplace=True)
        x_test.drop(columns=columns_to_drop, inplace=True)

    return x_train, x_val, x_test, y_train, y_val, y_test


def process_data_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Processes the DataFrame for inference.
    '''
    logger.info("Applying common transformations...")
    df, binary_cols, categorical_cols, numerical_cols = apply_common_transformations(df)
    logger.info(f"Dropping categorical columns: {categorical_cols}...")
    df.drop(columns=categorical_cols, inplace=True)
    df.drop(columns=['outcome'], inplace=True)

    return df


def apply_common_transformations(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list, list]:
    '''
    Applies the transformations to the DataFrame.
    '''

    df.drop(columns=['variant_id', 'order_id', 'user_id', 'created_at', 'order_date'], inplace=True)

    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    categorical_cols = [col for col in df.select_dtypes(include='object'
                                                        ).columns if col not in binary_cols]
    numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64']
                                                      ).columns if col not in binary_cols and col not in categorical_cols]

    return df, binary_cols, categorical_cols, numerical_cols


def train_test_val_split(df: pd.DataFrame, train_size: float = 0.7,
                         val_size: float = 0.2, test_size: float = 0.1
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Performs a time-split on the DataFrame to make a 3-way split
    '''

    n = len(df)

    potential_train_rows = df[:int(n * train_size)]
    train_th_day = potential_train_rows['order_date'].iloc[-1]
    train_data = df[df['order_date'] <= train_th_day]
    y_train = train_data['outcome']
    x_train = train_data.drop(columns=['outcome'])

    potential_val_rows = df[int(n * train_size):int(n * (train_size + val_size))]
    val_th_day = potential_val_rows['order_date'].iloc[-1]
    val_data = df[(df['order_date'] > train_th_day) & (df['order_date'] <= val_th_day)]
    y_val = val_data['outcome']
    x_val = val_data.drop(columns=['outcome'])

    test_data = df[df['order_date'] > val_th_day]
    y_test = test_data['outcome']
    x_test = test_data.drop(columns=['outcome'])

    return x_train, x_val, x_test, y_train, y_val, y_test

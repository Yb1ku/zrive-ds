from typing import Union, Tuple
import pandas as pd


def preprocess_data(data: pd.DataFrame,
                    min_items: int = 5,
                    train: bool = True) -> pd.DataFrame:
    '''Preprocess the input data before splitting.'''

    if train:
        data = data.loc[data.groupby('order_id')
                        .filter(lambda x: (x.outcome == 1).sum() >= min_items).index]
    data.sort_values(by='created_at')

    return data


def train_test_val_split(df: pd.DataFrame,
                         train_size: float = 0.7,
                         val_size: float = 0.2,
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                    pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    '''Time split the data into train, validation, and test sets.'''

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


def remove_info_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str],
                                                   list[str], list[str]]:
    '''Remove columns regarding information about ids and time.'''

    df.drop(columns=['variant_id', 'order_id', 'user_id', 'created_at', 'order_date'], inplace=True)

    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    categorical_cols = [col for col in df.select_dtypes(include='object')
                        .columns if col not in binary_cols]
    numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64'])
                      .columns if col not in binary_cols and col not in categorical_cols]

    return df, binary_cols, categorical_cols, numerical_cols


def remove_not_useful_columns(df: pd.DataFrame,
                              remove_categorical: bool = True) -> pd.DataFrame:
    '''Remove columns which are not useful for the model.'''

    df.drop(columns=[
        'count_adults', 'count_children', 'count_pets', 'people_ex_baby',
        'days_since_purchase_variant_id', 'days_since_purchase_product_type'
    ], inplace=True)
    if remove_categorical:
        categorical_cols = df.select_dtypes(include='object').columns
        df.drop(columns=categorical_cols, inplace=True)

    return df


def full_preprocessing_pipeline(df: pd.DataFrame,
                                min_items: int = 5,
                                train: bool = True
                                ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                 pd.DataFrame, pd.DataFrame, pd.DataFrame],
                                           pd.DataFrame]:
    '''Full preprocessing pipeline.'''

    df = preprocess_data(df, min_items=min_items, train=False)
    if not train:
        df, _, _, _ = remove_info_columns(df)
        df = remove_not_useful_columns(df)
        return df
    x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_split(df)

    def _clean_data(split_df: pd.DataFrame) -> pd.DataFrame:
        split_df, _, _, _ = remove_info_columns(split_df)
        split_df = remove_not_useful_columns(split_df)
        return split_df

    x_train = _clean_data(x_train)
    x_val = _clean_data(x_val)
    x_test = _clean_data(x_test)

    return x_train, x_val, x_test, y_train, y_val, y_test

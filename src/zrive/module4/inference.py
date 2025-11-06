import pandas as pd
import logging
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
from data_processing import process_data_for_inference
from typing import Any


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] - %(funcName)s - %(message)s')


def convert_request_to_dataframe(req: Any) -> tuple[pd.DataFrame, list]:
    '''
    Converts the incoming request to a pandas DataFrame.

    :param req: Incoming request data (can be dict or list of dicts).
    :type req: Any

    :returns:
        - **df** (*pd.DataFrame*) - DataFrame constructed from the request data.
        - **user_ids** (*list*) - List of user IDs extracted from the DataFrame
    '''

    logger.info("Converting request dictionary to DataFrame...")
    if isinstance(req, dict):
        first_val = next(iter(req.values()))
        if isinstance(first_val, dict):
            df = pd.DataFrame.from_dict(req, orient='index')
            df.index.name = 'user_id'
            df = df.reset_index()
            user_ids = df['user_id'].tolist()
        else:
            df = pd.DataFrame(req)
            user_ids = df['user_id'].tolist() if 'user_id' in df.columns else df.index.tolist()
    else:
        df = pd.DataFrame(req)
        user_ids = df['user_id'].tolist() if 'user_id' in df.columns else df.index.tolist()

    logger.info(f"Loaded DataFrame with shape: {df.shape}; extracted {len(user_ids)} user_ids.")
    return df, user_ids


def load_model(model_path: Path) -> Pipeline:
    '''
    Loads a saved model from the specified path.

    :param model_path: Path to the saved model file.
    :type model_path: Path

    :returns:
        - **model** (*sklearn.pipeline.Pipeline*) - The loaded model.
    '''

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    return model


def predict(model: Pipeline, df: pd.DataFrame, user_ids: list) -> dict:
    '''
    Makes a prediction using the provided model and DataFrame.

    :param model: Trained sklearn Pipeline model.
    :type model: Pipeline
    :param df: Input DataFrame for prediction.
    :type df: pd.DataFrame
    :param user_ids: List of user IDs corresponding to the DataFrame rows.
    :type user_ids: list

    :returns:
        - **json_response** (*dict*) - Dictionary containing status code and predictions.
    '''
    logger.info("Making predictions...")
    processed_df = process_data_for_inference(df)
    predictions = model.predict(processed_df)
    logger.info("Predictions made successfully.")

    json_predictions = {}
    for user in range(len(predictions)):
        json_predictions[user_ids[user]] = predictions[user]

    json_response = {
        'statusCode': 200,
        'body': json_predictions
    }

    return json_response


if __name__ == "__main__":
    # Example usage
    data_path = 'data/feature_frame.csv'
    df = pd.read_csv(data_path)
    req = df.iloc[0:3].to_dict(orient='records')

    req_df, user_ids = convert_request_to_dataframe(req)
    model_path = Path('models/push_2025_10_27.joblib')
    model = load_model(model_path)
    predictions = predict(model, req_df, user_ids)
    print(predictions)

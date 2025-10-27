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
    Converts a request (dict or list of record dicts) to a
    pandas DataFrame and returns a list of user_ids.
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
    '''

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    return model


def predict(model: Pipeline, df: pd.DataFrame, user_ids: list) -> dict:
    '''
    Makes a prediction using the provided model and DataFrame.
    '''
    try:
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
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        json_response = {
            'statusCode': 500,
            'body': f"Error during prediction: {e}"
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

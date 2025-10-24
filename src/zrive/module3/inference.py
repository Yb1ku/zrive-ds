import os
import joblib
import typing
import pandas as pd
from zrive.module3.data import full_preprocessing_pipeline


model_dir = 'models/'


def load_model(commit_hash: str):
    '''Load a model given its commit hash.'''
    model_name = f'logistic_regression_{commit_hash}.pkl'
    model_path = os.path.join(model_dir, model_name)
    model = joblib.load(model_path)
    return model


def predict(model: typing.Any, x: pd.DataFrame) -> pd.Series:
    '''Make the inference using the loaded model and preprocessed data.'''
    processed_data = full_preprocessing_pipeline(x, train=False)
    y_pred = model.predict(processed_data)

    return y_pred

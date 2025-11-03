import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime
import logging
from pathlib import Path
from data_processing import preprocess_data_for_training
import numpy as np
from matplotlib.figure import Figure
from typing import Optional


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] - %(funcName)s - %(message)s')


def create_pipeline(model_params: dict, seed: int = 42) -> Pipeline:
    '''
    Creates the sklearn Pipeline with an XGBoost classifier.
    '''

    logger.info("Creating the model pipeline...")
    model = Pipeline(steps=[
        ('classifier', XGBClassifier(**model_params, random_state=seed))
    ])
    return model


def fit_model(data_path: Path, model_path: Path, model_params: dict,
              name: str = 'auto', save_figure: bool = False, seed: int = 42,
              columns_to_drop: Optional[list[str]] = None,
              ) -> tuple[Figure, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Fits the model and saves it along with its performance plots.
    '''

    logger.info("Loading data...")
    data = pd.read_csv(data_path)

    logger.info("Preprocessing data...")
    x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data_for_training(data,
                                                                                  columns_to_drop=columns_to_drop)

    model = create_pipeline(model_params, seed=seed)
    logger.info("Training the model...")
    model.fit(x_train, y_train)

    y_train_probs = model.predict_proba(x_train)[:, 1]
    y_val_probs = model.predict_proba(x_val)[:, 1]

    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_probs)
    fpr_val,   tpr_val,   _ = roc_curve(y_val,   y_val_probs)
    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_val = auc(fpr_val,   tpr_val)

    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_probs)
    precision_val,   recall_val,   _ = precision_recall_curve(y_val,   y_val_probs)

    ap_train = average_precision_score(y_train, y_train_probs)
    ap_val = average_precision_score(y_val,   y_val_probs)

    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr_train, tpr_train, label=f'Train ROC AUC: {roc_auc_train:.2f}')
    plt.plot(fpr_val,   tpr_val,   label=f'Val ROC AUC: {roc_auc_val:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall_train, precision_train, label=f'Train AP: {ap_train:.2f}')
    plt.plot(recall_val,   precision_val,   label=f'Val AP: {ap_val:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.suptitle('Model Performance on Train and Val sets')
    plt.tight_layout()

    if save_figure:
        logger.info(f"Saving performance plots in {model_path}...")
        os.makedirs(model_path, exist_ok=True)
        plot_path = os.path.join(model_path, f'push_{DATE}_performance.png')
        plt.savefig(plot_path)

    plt.close()

    logger.info(f"Saving the trained model in {model_path}...")
    if name == 'auto':
        joblib.dump(model, os.path.join(model_path, f'push_{DATE}.joblib'))

    logger.info("Model training and saving completed.")

    return fig, recall_train, precision_train, recall_val, precision_val


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_PATH = BASE_DIR / "data" / "feature_frame.csv"
    MODEL_PATH = BASE_DIR / "models"
    DATE = datetime.today().strftime('%Y_%m_%d')
    RANDOM_SEED = 42
    model_params = {
        'n_estimators': 300,
        'eta': 0.01,
        'gamma': 5,
        'max_depth': 8
    }

    fit_model(DATA_PATH, MODEL_PATH, model_params)

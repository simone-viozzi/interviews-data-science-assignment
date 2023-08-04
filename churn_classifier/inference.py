from typing import Any
from joblib import load
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
from churn_classifier.data_cleaning import clean_dataset


class DatasetRow(BaseModel):
    city: str
    city_development_index: float
    gender: str | None
    relevent_experience: str
    enrolled_university: str | None
    education_level: str | None
    major_discipline: str | None
    experience: str | None
    company_size: str | None
    company_type: str | None
    last_new_job: str | None
    training_hours: float


class ChurnClassifier:
    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        self.model: Pipeline = load(model_path / "model.joblib")

    def df_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = clean_dataset(X)
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        return pd.DataFrame(
            {
                "prediction": y_pred,
                "prediction_proba": y_pred_proba,
            },
            index=X.index,
        )


    def predict(self, X: DatasetRow) -> float:
        X_df = pd.DataFrame([X.model_dump()])
        X_df = clean_dataset(X_df)

        # get only the prediction probability & assume only one row
        return self.model.predict_proba(X_df)[0, 1]
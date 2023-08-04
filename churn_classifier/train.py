import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
from joblib import dump
from churn_classifier.data_cleaning import clean_dataset


from churn_classifier.classification_evaluation import classification_eval

default_categorical_columns = (
    "city",
    "gender",
    "enrolled_university",
    "major_discipline",
    "company_type",
)


def train(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: tuple[str, ...] = default_categorical_columns,
    save_model_path: str | Path,
):
    save_model_path = Path(save_model_path)

    X = clean_dataset(X).drop(columns=["enrollee_id"])

    categorical_preprocessor = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    preprocessor = ColumnTransformer(
        [("categorical", categorical_preprocessor, categorical_columns)],
        remainder="passthrough",
    )

    model = make_pipeline(preprocessor, HistGradientBoostingClassifier())
    cv_result = cross_validate(model, X, y, cv=5, return_estimator=True)

    scores = cv_result["test_score"]
    print("The mean cross-validation accuracy is: " f"{scores.mean():.3f} ± {scores.std():.3f}")

    model = cv_result["estimator"][scores.argmax()]

    y_pred = model.predict(X)

    classification_eval(
        y_test=y,
        y_pred=y_pred,
    )

    dump(model, (save_model_path / "model.joblib").as_posix())

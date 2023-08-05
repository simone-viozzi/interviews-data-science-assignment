import tempfile
from pathlib import Path

import pandas as pd

from churn_classifier.data_cleaning import clean_dataset
from churn_classifier.dataset import default_dataset
from churn_classifier.inference import ChurnClassifier, DatasetRow
from churn_classifier.train import train
from churn_classifier.utils import get_X_y


def train_model(tmpdir):
    dataset = clean_dataset(default_dataset.copy())
    X, y = get_X_y(dataset)

    train(
        X=X,
        y=y,
        save_model_path=tmpdir,
    )


def inference_test():
    inference_dataser = default_dataset.copy().drop(columns=["target"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        train_model(tmpdir)

        model = ChurnClassifier(tmpdir)

        # test the df_predict method
        print(model.df_predict(inference_dataser.iloc[:6]))

        # get a single row
        row = inference_dataser.iloc[0].to_dict()

        # replace nan with None
        row = {k: None if pd.isna(v) else v for k, v in row.items()}

        # convert to a DatasetRow
        row_dataset = DatasetRow(**row)  # type: ignore

        # test the predict method
        print(model.predict(row_dataset))


if __name__ == "__main__":
    inference_test()

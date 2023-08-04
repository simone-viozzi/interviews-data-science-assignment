import pandas as pd
from churn_classifier.utils import get_X_y
from churn_classifier.train import train
from churn_classifier.dataset import default_dataset
import tempfile
from pathlib import Path

def train_test():
    
    X = default_dataset.copy().drop(columns=["target"])
    y = default_dataset["target"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        train(
            X=X,
            y=y,
            save_model_path=tmpdir,
        )

        if not (tmpdir / "model.joblib").exists():
            msg = "Model not saved"
            raise AssertionError(msg)


if __name__ == "__main__":
    train_test()

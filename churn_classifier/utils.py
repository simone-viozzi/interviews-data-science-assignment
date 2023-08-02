import pandas as pd

def get_X_y(dataset: pd.DataFrame):
    data = dataset.copy()

    X = data.drop(columns=["enrollee_id", "target"])
    y = data["target"]

    return X, y
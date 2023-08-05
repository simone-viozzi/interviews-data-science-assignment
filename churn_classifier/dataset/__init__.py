import importlib.resources as pkg_resources

import pandas as pd

with pkg_resources.open_text("churn_classifier.dataset", "churn.csv") as f:
    default_dataset = pd.read_csv(f)

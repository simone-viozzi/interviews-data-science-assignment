import pandas as pd
import importlib.resources as pkg_resources

with pkg_resources.open_text("churn_classifier.dataset", "churn.csv") as f:
    default_dataset = pd.read_csv(f)

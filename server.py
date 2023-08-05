import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile

from churn_classifier.dataset import default_dataset
from churn_classifier.inference import ChurnClassifier, DatasetRow
from churn_classifier.train import train

app = FastAPI()

# if the env var MODEL_PATH is not set, use a temp dir
if "MODEL_PATH" in os.environ:
    models_path = Path(os.environ["MODELS_PATH"])
else:
    models_path = Path(tempfile.TemporaryDirectory().name)

if not models_path.exists():
    models_path.mkdir(parents=True)


@app.get("/ping")
async def ping():
    return "pong"


@app.post("/train")
async def train_endpoint(dataset: UploadFile | None = None):
    df = default_dataset if dataset is None else pd.read_csv(dataset.file)

    X = df.copy().drop(columns=["target"])
    y = df["target"]

    model_id = uuid.uuid4()

    model_path = models_path / model_id.hex
    model_path.mkdir(parents=True)

    print(f"Saving model to {model_path}")

    train(
        X=X,
        y=y,
        save_model_path=model_path,
    )

    return {"model_id": model_id.hex}


@app.delete("/cleanup")
async def cleanup():
    models = list(models_path.iterdir())

    for model in models:
        shutil.rmtree(model)

    return {"message": f"Deleted {len(models)} models"}


@app.post("/models/{model_id}")
async def predict(model_id: str, row: DatasetRow):
    model_path = models_path / model_id

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    model = ChurnClassifier(model_path=model_path)
    return model.predict(row)


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    model_path = models_path / model_id
    shutil.rmtree(model_path)
    return {"message": f"Deleted model {model_id}"}


@app.get("/models")
async def get_models():
    models = list(models_path.iterdir())
    return {"models": [model.name for model in models]}

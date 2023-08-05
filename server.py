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

# if the env var MODELS_PATH is not set, use a temp dir
if "MODELS_PATH" in os.environ:
    models_path = Path(os.environ["MODELS_PATH"])
else:
    models_path = Path(tempfile.TemporaryDirectory().name)

if not models_path.exists():
    models_path.mkdir(parents=True)


tags_metadata = [
    {
        "name": "train",
        "description": "Train a new model",
    },
    {
        "name": "predict",
        "description": "Predict whether a customer will churn",
    },
    {
        "name": "models",
        "description": "Manage models",
    },
]

app = FastAPI(openapi_tags=tags_metadata)


@app.get("/ping")
async def ping():
    return "pong"


@app.post(
    "/train",
    tags=["train"],
    summary="Train a new model",
    description=(
        "Train a new model using the default dataset or a dataset uploaded as a CSV file.\n\n"
        "The dataset is assumed to have the same format as the default dataset.\n\n"
        "The model is saved to disk and can be used for predictions using the returned model ID."
    ),
)
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


@app.post(
    "/models/{model_id}",
    tags=["predict"],
    summary="Predict whether a customer will churn",
    description=(
        "Predict whether a customer will churn using a model trained by the `/train` endpoint.\n\n"
        "This endpoint expects a JSON object in the body with the same format "
        "as the default dataset, minus the `target` column."
    ),
)
async def predict(model_id: str, row: DatasetRow):
    model_path = models_path / model_id

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    model = ChurnClassifier(model_path=model_path)
    return model.predict(row)


@app.get(
    "/models",
    tags=["models"],
    summary="List all available models",
)
async def get_models():
    models = list(models_path.iterdir())
    return {"models": [model.name for model in models]}


@app.delete(
    "/models/{model_id}",
    tags=["models"],
    summary="Delete a model by ID",
)
async def delete_model(model_id: str):
    model_path = models_path / model_id
    shutil.rmtree(model_path)
    return {"message": f"Deleted model {model_id}"}


@app.delete(
    "/cleanup",
    tags=["models"],
    summary="Delete all models",
)
async def cleanup():
    models = list(models_path.iterdir())

    for model in models:
        shutil.rmtree(model)

    return {"message": f"Deleted {len(models)} models"}

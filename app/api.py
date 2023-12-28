import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4
from loguru import logger

from api.persistence.data_access import PredictionPersistence, ModelType
from api.persistence.core import get_db_session
from app import __version__
from app.config import settings
from app.schemas.health import Health
from app.schemas.predict import MultipleDataInputs, PredictionResults
from model import __version__ as model_version
from model.predict import make_prediction

api_router = APIRouter()

# A dictionary to store the results of background tasks
background_tasks_results = {}

@api_router.get("/health", response_model=Health, status_code=200)
def health() -> dict:
    health = Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()

async def make_save_predictions_background(task_id: str, db_session: Session, db_model: ModelType, input_data: list):
    prediction_persistence = PredictionPersistence(db_session=db_session)
    result = prediction_persistence.make_save_predictions(db_model=db_model, input_data=input_data)
    # Store the result in the dictionary using the task ID
    background_tasks_results[task_id] = result

@api_router.post("/predict", status_code=200)
async def predict(input_data: MultipleDataInputs, background_tasks: BackgroundTasks, db_session: Session = Depends(get_db_session)) -> Any:
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    input_data = input_df.replace({np.nan: None}).to_dict(orient="records")

    # Generate a unique task ID
    task_id = str(uuid4())

    # Add background task
    background_tasks.add_task(make_save_predictions_background, task_id, db_session, ModelType.LASSO, input_data)

    # Return the task ID
    return {"task_id": task_id}

@api_router.get("/predict/{task_id}", response_model=PredictionResults, status_code=200)
async def get_prediction_result(task_id: str) -> Any:
    # Retrieve the result using the task ID
    result = background_tasks_results.get(task_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return result


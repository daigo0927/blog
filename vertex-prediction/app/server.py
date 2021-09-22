import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from enum import Enum
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel


# AIP_STORAGE_URI = os.environ.get('AIP_STORAGE_URI')
# AIP_HTTP_PORT = os.environ.get('AIP_HTTP_PORT', '80')
AIP_HEALTH_ROUTE = os.environ.get('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE = os.environ.get('AIP_PREDICT_ROUTE', '/predict')

app = FastAPI()
model = lgb.Booster(model_file='model.lgb')


class Specie(Enum):
    ADELIE = 0
    CHINSTRAP = 1
    GENTOO = 2


class PenguinFeature(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float


class Parameters(BaseModel):
    return_confidence: bool

    
class Prediction(BaseModel):
    specie: str
    confidence: Optional[float]


class Predictions(BaseModel):
    predictions: List[Prediction]


@app.get(AIP_HEALTH_ROUTE, status_code=200)
async def health():
    return {'health': 'ok'}


@app.post(AIP_PREDICT_ROUTE,
          response_model=Predictions,
          response_model_exclude_unset=True)
async def predict(instances: List[PenguinFeature],
                  parameters: Optional[Parameters] = None):
    instances = pd.DataFrame([x.dict() for x in instances])
    preds = model.predict(instances)

    indices = np.argmax(preds, axis=-1)
    confidences = np.max(preds, axis=-1)

    if parameters is not None:
        return_confidence = parameters.return_confidence
    else:
        return_confidence = False

    outputs = []
    for index, confidence in zip(indices, confidences):
        specie = Specie(index).name
        if return_confidence:
            outputs.append(Prediction(specie=specie, confidence=confidence))
        else:
            outputs.append(Prediction(specie=specie))
            
    return Predictions(predictions=outputs)
    

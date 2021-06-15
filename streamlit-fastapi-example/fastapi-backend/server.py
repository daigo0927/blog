import sys
import time
import asyncio
from fastapi import FastAPI, BackgroundTasks, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict
from logging import getLogger

logger = getLogger(__name__)

app = FastAPI()


@app.get('/')
async def index():
    return 'index'


def predict_image(t: int) -> None:
    for i in range(t):
        sys.stdout.write(f'\r{i+1}')
        sys.stdout.flush()
        time.sleep(1)
    print()


@app.post('/predict')
async def predict(files: List[UploadFile] = File(...),
                  background_tasks: BackgroundTasks = None):
    background_tasks.add_task(predict_image, 3*len(files))
    return f'{len(files)} files are submitted!'


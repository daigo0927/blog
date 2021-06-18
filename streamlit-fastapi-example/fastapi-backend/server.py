import os
import sys
import time
import asyncio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Dict
from datetime import datetime
from logging import getLogger
from google.cloud import storage
from fastapi import FastAPI, BackgroundTasks, File, UploadFile
from fastapi.logger import logger
from tensorflow.keras.applications.efficientnet import EfficientNetB0, decode_predictions


BUCKET_NAME = os.environ.get('BUCKET_NAME')
IMAGE_SIZE = (224, 224)

gunicorn_error_logger = getLogger("gunicorn.error")
gunicorn_logger = getLogger("gunicorn")
uvicorn_access_logger = getLogger("uvicorn.access")
uvicorn_access_logger.handlers = gunicorn_error_logger.handlers
logger.handlers = gunicorn_error_logger.handlers
logger.setLevel(gunicorn_logger.level)

plt.switch_backend('Agg')

app = FastAPI()

model = EfficientNetB0(weights=None)
model.load_weights('weights/effnet-b0.ckpt')


def save_prediction(image: np.ndarray,
                    classes: List[str],
                    probs: List[float],
                    savename: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title('Input image')
    ax1.imshow(image)
    ax2.set_title('Top probabilities')
    ax2.barh(classes, probs)
    ax2.invert_yaxis()
    fig.tight_layout()
    plt.savefig(savename)


def predict_images(files: List[UploadFile], job_id: str) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    for file in files:
        image = tf.io.decode_image(file.file.read())
        image = tf.image.resize_with_pad(image, *IMAGE_SIZE)
        pred = model.predict(image[None])
        pred = decode_predictions(pred)[0]

        image = image.numpy().astype(np.uint8)
        _, classes, probs = list(zip(*pred))

        filename = f'pred_{file.filename}'
        save_prediction(image, classes, probs, savename=filename)
        
        blob = bucket.blob(f'results/{job_id}/{filename}')
        blob.upload_from_filename(filename)


@app.post('/predict')
async def predict(files: List[UploadFile] = File(...),
                  background_tasks: BackgroundTasks = None):
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    background_tasks.add_task(predict_images, files=files, job_id=job_id)
    return f'{len(files)} files are submitted'


@app.get('/results')
async def results(job_id: str = None, limit=5):
    client = storage.Client()
    prefix = f'results/{job_id}' if job_id is not None else 'results'
    blobs = client.list_blobs(bucket_or_name=BUCKET_NAME, prefix=prefix)
    blob_names = [b.name for b in blobs]
    return blob_names

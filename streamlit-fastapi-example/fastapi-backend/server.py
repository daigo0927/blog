import os
import sys
import time
import asyncio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, File, UploadFile
from tensorflow.keras.applications.efficientnet import EfficientNetB0, decode_predictions


IMAGE_SIZE = (224, 224)

plt.switch_backend('Agg')

app = FastAPI()

model = EfficientNetB0(weights=None)
model.load_weights('weights/effnet-b0.ckpt')


def save_prediction(image: np.ndarray,
                    classes: List[str],
                    probs: List[float],
                    savepath: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title('Input image')
    ax1.imshow(image)
    ax2.set_title('Top probabilities')
    ax2.barh(classes, probs)
    ax2.invert_yaxis()
    fig.tight_layout()
    plt.savefig(savepath)


def predict_images(files: List[UploadFile], job_id: str) -> None:
    savedir = Path(f'./results/{job_id}')
    if not savedir.exists():
        savedir.mkdir(parents=True)
    
    for file in files:
        image = tf.io.decode_image(file.file.read())
        image = tf.image.resize_with_pad(image, *IMAGE_SIZE)
        pred = model.predict(image[None])
        pred = decode_predictions(pred)[0]

        image = image.numpy().astype(np.uint8)
        _, classes, probs = list(zip(*pred))

        savepath = savedir/file.filename
        save_prediction(image, classes, probs, savepath=savepath)


@app.post('/predict')
async def predict(files: List[UploadFile] = File(...),
                  background_tasks: BackgroundTasks = None):
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    background_tasks.add_task(predict_images, files=files, job_id=job_id)
    return f'{len(files)} files are submitted'


@app.get('/results')
async def results():
    p = Path('results')
    # results/yyyymmdd_hhmmss/(png|jpg)
    result_files = [str(pp) for pp in p.glob('*/*')]
    return result_files

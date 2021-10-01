from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from tasks import celery, calc_bmi

app = FastAPI()


class TaskStatus(BaseModel):
    id: str
    status: Optional[str]
    result: Optional[float]


@app.post('/bmi', response_model=TaskStatus, response_model_exclude_unset=True)
def calculate_bmi(weight: float, height: float):
    task = calc_bmi.delay(weight, height)
    return TaskStatus(id=task.id)


@app.get('/bmi/{task_id}', response_model=TaskStatus)
def check_status(task_id: str):
    result = celery.AsyncResult(task_id)
    status = TaskStatus(id=task_id, status=result.status, result=result.result)
    return status

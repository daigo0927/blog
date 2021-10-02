# Asynchronous task queue using Celery

Sample asynchronous task queue application using Celery.

The sample app is composed of below components:

- **FastAPI** provides APIs to post BMI calculation task and check its status
- **Celery** manages tasks
- **Flower** provides a dashboard for monitors the status of tasks and workers
- **Redis** stores task queue as a message broker and also task results as a backend

# Usage

## Launch containers

``` shell
# Build images
docker compose build

# Run containers
docker compose up

# Stop containers
docker compose down
```

## Submit task

``` shell
curl -X 'POST' \
  'http://127.0.0.1:8080/bmi' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"weight": 65, "height": 1.8}'
# >> {"id":<task_id>}
```

## Check task status

``` shell
curl 'http://127.0.0.1:8080/bmi/<task_id>'
# after task succeeded
# >> {"id":<task_id>,"status":"SUCCESS","result":19.031141868512112}
```

The status varies with the task processing like `FAILURE`, `STARTED`, etc.

## Other features

FastAPI provides API docs at `http://127.0.0.1:8080/docs` and you can try APIs on your browser.

Flower (Celery monitaring tool) provides a dashboard for monitoring workers and tasks. You can see the dashboard at `http://127.0.0.1:5556`


# References

- [Celery](https://docs.celeryproject.org/en/stable/index.html#)
- [Flower](https://flower.readthedocs.io/en/latest/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [https://github.com/testdrivenio/fastapi-celery](https://github.com/testdrivenio/fastapi-celery)

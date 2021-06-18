FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

# Install Poetry
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /app/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --no-dev

COPY . /app/
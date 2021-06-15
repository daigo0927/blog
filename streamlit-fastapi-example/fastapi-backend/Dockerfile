FROM python:3.8-slim
EXPOSE 80
WORKDIR /app

RUN pip install poetry
COPY poetry.lock pyproject.toml .
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
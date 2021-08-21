FROM python:3.8-slim
WORKDIR /root

RUN pip install poetry
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . .

ENTRYPOINT ["python", "main.py"]
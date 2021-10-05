FROM python:3.8-slim
WORKDIR /root

RUN apt-get update -y \
    && apt-get install -y libgomp1 \
    && apt-get -y clean all

RUN pip install poetry
COPY poetry.lock pyproject.toml .
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . .

ENTRYPOINT ["python", "main.py"]
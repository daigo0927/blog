FROM python:3.8-slim
WORKDIR /root

# Upgrade pip to latest
RUN pip install -U pip \
    && pip install -U google-cloud-aiplatform google-cloud-storage

COPY main.py ./main.py

ENTRYPOINT ["python", "main.py"]
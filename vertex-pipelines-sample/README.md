# ML workflow on Vertex Pipelines

## Directry overview

```
vertex-pipelines-sample
├── README.md
├── components
│   ├── evaluate
│   │   ├── Dockerfile
│   │   ├── component.yaml
│   │   ├── main.py
│   │   ├── poetry.lock
│   │   └── pyproject.toml
│   ├── preprocess
│   │   ├── .dockerignore
│   │   ├── Dockerfile
│   │   ├── component.yaml
│   │   ├── main.py
│   │   ├── poetry.lock
│   │   ├── pyproject.toml
│   └── train
│       ├── .dockerignore
│       ├── Dockerfile
│       ├── component.yaml
│       ├── main.py
│       ├── poetry.lock
│       └── pyproject.toml
├── docker-compose.yaml
├── pipeline.py
└── requirements.txt

```

## Components

This pipeline is composed of 3 components: preprocess, train, evaluate.
Exact settings are defined at `components/(preprocess,train,evaluate)/component.yaml`

### Preprocess

Drop unnecessary columns and split raw csv into train/val files. I use [Palmer Penguins dataset](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv) and locate it on `src_path` of GCS (i.e. `gs://<bucket>/penguins.csv`).

- inputs:
  - `src_path`: Path to the raw CSV file (assuming GCS)
  - `n_splits`: Split of train/val set. (n-1)/n is assigned to train, 1/n is to validation
- outputs:
  - `train_path`: Path to the output train file
  - `val_path`: Path to the output val file

### Train

Train LightGBM using train set and early-stop on validation set.

- inputs:
  - `train_path`: Path to the train file
  - `val_path`: Path to the val file
  - `learning_rate`: Learning rate for LightGBM training
  - `max_depth`: Maximum depth for single tree
- outputs:
  - `model_path`: Path to the resulting model file

### Evaluate

Evaluate the trained LightGBM and visualize feature importance.

- inputs:
  - `val_path`: Path to the val file
  - `model_path`: Model file for evaluation
- outputs:
  - `mlpipeline_metrics`: Path to the metrics file
  - `visualize_path`: Path to the visualization file
  
## Setup

### GCS Bucket

Create 2 GCS buckets:

- Source dataset bucket: for locating the raw CSV (Palmer Penguins dataset). Replace the `src_path` of L43@pipeline.py
- Pipeline root bucket: for locating the pipeline artifacts. Replace the `pipeline_root` of L36@pipeline.py

Note that bucket name must be globally unique. Use different bucket names from mine.

### Docker images

``` bash
# set GCP project id as environment variable
export GCP_PROJECT_ID=<gcp-project-id>

# Build images and push into container repository
docker compose build
docker compose push
```

### Pipeline

``` bash
# install build dependencies (virtualenv recommended)
pip install -r requirements.txt

# build pipeline, output vertex-pipelines-sample.json
python pipeline.py
```

## Run

Upload the generated pipeline JSON file in Cloud Console. Put required parameters (in this sample, project_id and region). Submit the pipeline execution.

## Notes

Currently (2021/08/21) Vertex Pipelines is available as a **preview** version service. There were a few points I have not figured out;

- Output file extension cannot be set: Variables with `outoutPath` type in each component are automatically set via Vertex Pipelines not via users. For example, `train_path` in preprocess component is treated as a path like `/gcs/path/to/train_path` in the script.
- Image file is not visible on Vertex Pipelines UI: Evaluation component visualize feature importance and save it as an artifact but not directory view in the UI.
- Metrics is no visible on Vertex Pipelines UI: Evaluation component writes accuracy as a metric but not visible the UI.

Though above notes, Vertex Pipelines and KFP are eagerly under development. I'm looking forward the updates.

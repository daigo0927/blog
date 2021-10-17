# End-to-End ML workflow on Vertex Pipelines

## Pipeline overview

The sample pipeline is composed of 4 components: preprocess, train, evaluate and deploy.

- **preprocess**: Preprocess source CSV file and split into train/val set
- **train**: Train LightGBM model on the processed sets
- **evaluation**: Evaluate the trained model on the val set
- **deploy**: Deploy the trained model on Vertex AI

Exact settings are defined at `components/(preprocess,train,evaluate,deploy)/component.yaml`.
In addition, `docker/serving` defines the container image to serve the prediction API, specified in the deploy process.

## Directry structure

```
vertex-pipelines-sample
├── .env(.dummy)
├── README.md
├── components
│   ├── deploy
│   │   ├── Dockerfile
│   │   ├── component.yaml
│   │   ├── main.py
│   │   ├── poetry.lock
│   │   └── pyproject.toml
│   ├── evaluate
│   │   └── ...
│   ├── preprocess
│   │   └── ...
│   └── train
│       └── ...
├── docker
│   └── serving
│       ├── Dockerfile
│       ├── server.py
│       ├── poetry.lock
│       └── pyproject.toml
├── docker-compose.yaml
├── pipeline.py
└── requirements.txt
```

# How to Run

Assuming you have a GCP account for running Vertex Pipelines and the related Google Cloud services.

## 1. Setup

### Create GCS buckets

This sample uses 2 Cloud Storage buckets (can be same). One for storing the original data CSV file, and another for the pipeline artifacts.

``` shell
gsutil mb -l <region> gs://<bucket1>
gsutil mb -l <region> gs://<bucket2>
```

### Put the original data source on the bucket

The sample pipeline uses [Palmer Penguins dataset](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv). Copy the `dataset/penguins.csv` to `<bucket1>`

```shell
gsutil cp ./dataset/penguins.csv gs://<bucket1>/penguins.csv
```

### Set environment variables

Write the common environment variables in `.env` file. This is reflected when building the containers and compiling the pipeline.

```shell
GCP_PROJECT_ID=<your GCP project id>
LOCATION=<GCP region same as 2 buckets>
SOURCE_CSV_URI=gs://<bucket1>/penguins.csv
ROOT_BUCKET=gs://<bucket2>
```

### Build and ship container images

Build containers and push them into Container Registry. This may take a few minutes.

```shell
docker compose build
docker compose push
```

## 2. Compile the pipeline

Install dependencies and compile the pipeline. I recommend to use python virtual environment (like pyenv).

```
pip install kfp==1.7.1 python-dotenv==0.19.1  # minimum dependencies
python pipeline.py  # compile
```

`pipeline.py` generates `vertex-pipelines-sample.json` which contains pipeline information.

## 3. Run the pipeline

Execute the pipeline via the generated `vertex-pipelines-sample.json` at Cloud Console.
Move [Vertex AI>Pipelines] on browser and [+CREATE RUN]. Choose the JSON as a pipeline file, set Run name, specify some pipeline parameters, and [SUBMIT] to launch the pipeline.

Submitted pipeline can be visualized on the browser like below:

![Generated pipeline](https://user-images.githubusercontent.com/23152884/137509509-05bd5a70-3d27-4a80-b3d4-a314c2770f77.png)

## Optional: Request prediction

This pipeline finally deploy the trained model to the Prediction Endpoint. The endpoint is ready to serve so that we can request a prediction. You can check the Model and Endpoint resources at [Vertex AI>Models] and [Vertex AI>Endpoints].

For example, prediction request via cURL is like:

``` shell
PROJECT_ID=`gcloud config list --format 'value(core.project)'`
LOCATION=<endpoint-region>
ENDPOINT_ID=<deployed-endpoint-id>  # Check from [Vertex AI>Endpoints]
INPUT_DATA_FILE=sample-request.json

curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/endpoints/${ENDPOINT_ID}:predict \
-d "@${INPUT_DATA_FILE}"
```

If you want to remove an endpoint resource, 1) undeploy tied models from the endpoint then 2) remove the endpoint. Undeployed model resource can be deployed to another endpoint.

# Component details

In Kubeflow Pipelines, each input/output type is defined in `component.yaml`. See the [official documentation](https://www.kubeflow.org/docs/components/pipelines/sdk/component-development/) for details.

## Preprocess

Drop unnecessary columns and split raw csv into train/val files.

- inputs:
  - `src_csv: String`: GCS path to the raw CSV file
  - `n_splits: Integer`: Split of train/val set. (n-1)/n is assigned to train, 1/n is to validation
- outputs:
  - `dataset: Dataset`: Processed dataset URI. In this example, this is a directory containing train/val.csv

## Train

Train LightGBM using train set and early-stop on validation set.

- inputs:
  - `dataset: Dataset`: Output from preprocess component
  - `learning_rate: Float`: Learning rate for LightGBM training
  - `max_depth: Integer`: Max tree depth
- outputs:
  - `artifact: Model`: Trained model artifacts. In this example, this is a directory containing `model.joblib` file.

## Evaluate

Evaluate the trained LightGBM.

- inputs:
  - `dataset: Dataset`: Output from preprocess component
  - `artifact: Model`: Output from train component
- outputs:
  - `metrics: Artifact`: Evaluation metrics artifacts. In this example, this is a directory containing metrics JSON file.
  
## Deploy

- inputs:
  - `artifact: Model`: Output from train component
  - parameters for configuring Vertex AI Model and Endpoint
- outputs: None
  

# Notes

Currently (2021/10/17) Vertex Pipelines is available as a **preview** version service. There were a few points I have not figured out:

- Output file extension cannot be set: Component variables with `outoutPath` type are automatically set via Vertex Pipelines not via users. For example, `artifact` variable in train component is treated as a path like `/gcs/<ROOT_BUCKET>/path/to/artifact` in the script. This makes a file type umbiguous, so I implement this as a directory and create actual files in it.
- It is unclear to use pipeline metrics and visualizations with container components: Though metrics and visualization utilities are introduced in the [Kubeflow Pipelines docs](https://www.kubeflow.org/docs/components/pipelines/sdk/pipelines-metrics/), it is not clear to use them in Vertex Pipelines.

Though above notes, Vertex Pipelines and KFP are eagerly under development. I'm looking forward to see more updates and samples :)

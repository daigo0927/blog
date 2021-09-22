# ML Model serving on Vertex AI

Sample resources for serving ML model (predicting penguin species) on Google Cloud Vertex AI.

Vertex AI offers 2 ways to serve ML model: 1) on pre-built containers, 2) on on custom containers. One can use pre-built containers if the model meets several requirements. Custom containers are more flexible way to serve ML model. See details in the [official documentation](https://cloud.google.com/vertex-ai/docs/general/import-model).

This example use custom containers for demonstrating the flexible model serving.

## Setup

``` shell
PROJECT_ID=`gcloud config list --format 'value(core.project)'`
```

In this example, uploading model and deployment is done via Cloud Console but you can also configure them by other ways (e.g. gcloud, Python SDK, etc.).

### 1. Prepare container

Build and push docker container where the model runs.

``` shell
docker image build ./app -t vertex-prediction
docker tag vertex-prediction gcr.io/${PROJECT_ID}/vertex-prediction:latest
docker image push gcr.io/${PROJECT_ID}/vertex-prediction:latest
```

(Optional) Run container locally

```
docker container run --rm -p 80:80 -e APP_MODULE=server:app --name vertex-pred vertex-prediction:latest
```

You can check the API docs on `localhost:80/docs` and try to post request for the `/predict` endpoint.

### 2. Upload model to Vertex AI

In this time, *model* is composed of the model artifact itself (trained LightGBM, DNN, etc.) and the container where the model runs.

On the cloud console, *model* can be uploaded via \[Vertex AI>Models>IMPORT]. Configuring:

1. Name and region
   - Region should be the same as the related bucket and following serving endpoint.
2. Model settings
   - Check \[Import an existing custom container\].
   - Custom container settings: select the pushed container images.
   - Environment variables: `APP_MODULE=server:app`, let Port be 80.
3. Explainability (optional)
   - None in this example.
   
Then \[IMPORT\] to import the model. This will take a few minutes to complete.

In this example, the trained model artifact is packaged into the container. Otherwise, you can locate an artifact on Cloud Storage and load it from a container. When you put the artifact location on \[Model settings>Custom container settings\], the container app can access it via `AIP_STORAGE_URI` environment variable. See details in the [official docs](https://cloud.google.com/vertex-ai/docs/general/import-model#aiplatform_upload_model_sample-gcloud)

### 3. Deploy to endpoint

Prediction can be served via creating endpoint and deploying the uploaded model to it.

On the Cloud Console, choose the uploaded model and select \[DEPLOY TO ENDPOINT\]:

1. Define your endpoint
   - Check *Create new endpoint* and set the endpoint name.
   - Loaction will be automatically set the same as the model.
   - Access: Standard
2. Model settings
   - Set minimum number of compute node (>=1) and machine-type as you want (single n1-standard-2 is recommended for the demo purpose).
3. Model monitoring
   - Keep default (disabled)
   
Then \[DEPLOY\] to deploy the model to the endpoint. After the deployment completed, prediction is served!

## Request predictions

Cloud Console provides a test field to request prediction to the deployed model. On \[Vertex AI>Models>(deployed model)>DEPLOY&TEST\], put a sample input and \[PREDICT\] then you can receive predictions. 

`sample-request.json` is a sample request contents. The model will return those categories and prediction confidence values. Generally, you will request predictions via REST API:

``` shell
ENDPOINT_ID=<deployed-endpoint-id>
INPUT_DATA_FILE=sample-request.json
```

when using curl:

```shell
curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://us-east1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-east1/endpoints/${ENDPOINT_ID}:predict \
-d "@${INPUT_DATA_FILE}"
```


# ML workflow example

Using GCP Vertex AI custom-job

## Run scripts directlly

``` bash
# set GCS credential
export GOOGLE_APPLICATION_CREDENTIALS=<gcs-credential-json>

# preprocess (at ./preprocess directory)
python main.py --csv-path gs://<dataset-bucket>/penguins.csv --output-dir .

# train (at ./train directory)
python main.py --datadir ../preprocess --out-bucket <train-bucket>
```

## Run as a container application

### Build

``` bash
# preprocess
docker image build ./preprocess -t workflow/preprocess:latest

# train
docker image build ./train -t workflow/train:latest
```

### Run locally

``` bash
# preprocess
docker container run -e GOOGLE_APPLICATION_CREDENTIALS=<gcs-credential-json> --name preprocess --rm workflow/preprocess:latest --csv-path gs://<source-bucket>/penguins.csv --output-dir gs://<preprocess-bucket>/<directory>

# train
docker container run -e GOOGLE_APPLICATION_CREDENTIALS=<gcs-credential-json> --name train --rm workflow/train:latest --datadir gs://<preprocess-bucket>/<directory> --out-bucket <train-bucket>
```

### Push to GCR

```
# preprocess
docker tag workflow/preprocess:latest gcr.io/<project-id>/workflow-example-preprocess:latest
docker push gcr.io/<project-id>/workflow-example-preprocess:latest

# train
docker tag workflow/train:latest gcr.io/<project-id>/workflow-example-train:latest
docker push gcr.io/<project-id>/workflow-example-train:latest
```

## Run as a custom job

``` bash
# run
gcloud ai custom-jobs create --region=<region> --display-name=<job-name> --config=<config-yaml>

# describe specific job state
gcloud ai custom-jobs describe <job-id> --region=<region>

# describe specific job via curl
curl -X GET -H "Authorization: Bearer "$(gcloud auth application-default print-access-token) https://<location>-aiplatform.googleapis.com/v1beta1/projects/<project-id>/locations/<location>/customJobs/<job-id>

# list jobs (filter succeeded jobs)
gcloud ai custom-jobs list --region=<region> --filter state=JOB_STATE_SUCCEEDED
```

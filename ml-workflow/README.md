# ML workflow example

Using GCP Vertex AI custom-job

## Local validation

### Build

``` bash
docker image build ./preprocess -t workflow/preprocess:latest
```

### Push to GCR

```
docker tag workflow/preprocess:latest gcr.io/<project-id>/workflow-example-preprocess:latest
docker push gcr.io/<project-id>/workflow-example-preprocess:latest
```

### Run locally

``` bash
# preprocess
docker container run -e GOOGLE_APPLICATION_CREDENTIALS=<GCS credential json> --name preprocess --rm workflow/preprocess:latest --job-dir . --csv-path gs://workflow-example-dataset/penguins.csv
```

### Run as a single custom-job

``` bash
gcloud ai custom-jobs create --region=<region> --display-name=<job-name> --config=<config-yaml>
```

import os
import json
import argparse
from google.cloud import aiplatform


def run(artifact_uri: str, model_name: str, serving_container_image_uri: str,
        serving_container_environment_variables: str,
        serving_container_ports: str, endpoint_name: str, deploy_name: str,
        machine_type: str, min_replicas: int, max_replicas: int, project: str,
        location: str):
    # convert the mounted /gcs/ pass to gs:// location
    artifact_uri = artifact_uri.replace('/gcs/', 'gs://', 1)

    # convert json string to dict
    if serving_container_environment_variables is not None:
        serving_container_environment_variables \
            = json.loads(serving_container_environment_variables)

    aiplatform.init(project=project, location=location)

    model = aiplatform.Model.upload(
        display_name=model_name,
        serving_container_image_uri=serving_container_image_uri,
        artifact_uri=artifact_uri,
        serving_container_environment_variables=
        serving_container_environment_variables,
        serving_container_ports=[serving_container_ports])

    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name={endpoint_name}', order_by='create_time desc')
    if len(endpoints) > 0:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    print(f'Target endpoint: {endpoint.resource_name}')

    model.deploy(endpoint=endpoint,
                 deployed_model_display_name=deploy_name,
                 machine_type=machine_type,
                 min_replica_count=min_replicas,
                 max_replica_count=max_replicas)
    print(model.display_name)
    print(model.resource_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy')
    parser.add_argument('--artifact-uri', type=str)
    # Model resource config
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--serving-container-image-uri', type=str)
    parser.add_argument('--serving-container-environment-variables', type=str)
    parser.add_argument('--serving-container-ports', type=int)
    # Endpoint config
    parser.add_argument('--endpoint-name', type=str)
    # Model Deployment config
    parser.add_argument('--deploy-name', type=str)
    parser.add_argument('--machine-type', type=str)
    parser.add_argument('--min-replicas', type=int)
    parser.add_argument('--max-replicas', type=int)
    # General config
    parser.add_argument('--project', type=str)
    parser.add_argument('--location', type=str)

    args = parser.parse_args()
    run(**vars(args))

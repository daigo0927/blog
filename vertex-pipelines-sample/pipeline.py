import os
from dotenv import load_dotenv
from kfp.v2 import dsl, compiler, components

load_dotenv('.env')
PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
LOCATION = os.environ.get('LOCATION')
SOURCE_CSV_URI = os.environ.get('SOURCE_CSV_URI')
ROOT_BUCKET = os.environ.get('ROOT_BUCKET')


@dsl.pipeline(name='vertex-pipelines-sample',
              description='Vertex Piplines sample',
              pipeline_root=ROOT_BUCKET)
def pipeline(learning_rate: float = 0.1, max_depth: int = 10) -> None:
    preprocess_op = components.load_component_from_file(
        'components/preprocess/component.yaml')
    preprocess_task = preprocess_op(src_csv=SOURCE_CSV_URI, n_splits=3)

    train_op = components.load_component_from_file(
        'components/train/component.yaml')
    train_task = train_op(dataset=preprocess_task.outputs['dataset'],
                          learning_rate=learning_rate,
                          max_depth=max_depth)
    train_task.custom_job_spec = {
        'displayName': train_task.name,
        'jobSpec': {
            'workerPoolSpecs': [{
                'containerSpec': {
                    'imageUri': train_task.container.image,
                    'args': train_task.arguments,
                },
                'machineSpec': {
                    'machineType': 'c2-standard-4'
                },
                'replicaCount': 1
            }],
        }
    }

    evaluate_op = components.load_component_from_file(
        'components/evaluate/component.yaml')
    _ = evaluate_op(dataset=preprocess_task.outputs['dataset'],
                    artifact=train_task.outputs['artifact'])

    deploy_op = components.load_component_from_file(
        'components/deploy/component.yaml')
    _ = deploy_op(
        artifact=train_task.outputs['artifact'],
        model_name='vp-sample-lightgbm',
        serving_container_image_uri=f'gcr.io/{PROJECT_ID}/vp-sample-serving',
        serving_container_environment_variables='{"APP_MODULE": "server:app"}',
        serving_container_ports=80,
        endpoint_name='vp-sample-endpoint',
        deploy_name='vp-sample-deploy',
        machine_type='n1-standard-2',
        min_replicas=1,
        project=PROJECT_ID,
        location=LOCATION)


compiler.Compiler().compile(pipeline_func=pipeline,
                            package_path='vertex-pipelines-sample.json')

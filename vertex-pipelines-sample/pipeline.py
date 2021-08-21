from kfp.v2 import dsl, compiler, components
from google_cloud_pipeline_components import experimental as gcc_exp


preprocess_op_raw = components.load_component_from_file(
    'components/preprocess/component.yaml'
)
preprocess_op = gcc_exp.run_as_vertex_ai_custom_job(
    component_spec=preprocess_op_raw,
    display_name='preprocess job',
    machine_type='e2-standard-4'
)

train_op_raw = components.load_component_from_file(
    'components/train/component.yaml'
)
train_op = gcc_exp.run_as_vertex_ai_custom_job(
    component_spec=train_op_raw,
    display_name='train job',
    machine_type='c2-standard-4'
)


@dsl.pipeline(
    name='vertex-pipelines-sample',
    description='Vertex Piplines sample',
    pipeline_root='gs://vertex-pipelines-sample'
)
def pipeline(project_id: str,
             region: str,
             learning_rate: float = 0.1,
             max_depth: int = 10) -> None:
    preprocess_task = preprocess_op(
        src_path='gs://workflow-example-dataset/penguins.csv',
        n_splits=3,
        gcp_project=project_id,
        gcp_region=region
    )
    _ = train_op(
        train_path=preprocess_task.outputs['train_path'],
        val_path=preprocess_task.outputs['val_path'],
        learning_rate=learning_rate,
        max_depth=max_depth,
        gcp_project=project_id,
        gcp_region=region
    )

    
compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path='vertex-pipelines-sample.json'
)

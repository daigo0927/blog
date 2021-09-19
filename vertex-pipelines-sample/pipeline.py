from kfp.v2 import dsl, compiler, components
# from google_cloud_pipeline_components import experimental as gcc_exp

# preprocess_op = gcc_exp.run_as_vertex_ai_custom_job(
#     component_spec=preprocess_op_raw,
#     display_name='preprocess-job',
#     machine_type='e2-standard-4'
# )


# train_op = gcc_exp.run_as_vertex_ai_custom_job(
#     component_spec=train_op_raw,
#     display_name='train-job',
#     machine_type='c2-standard-4'
# )


# evaluate_op = gcc_exp.run_as_vertex_ai_custom_job(
#     component_spec=evaluate_op_raw,
#     display_name='evaluate-job',
#     machine_type='e2-standard-4'
# )


@dsl.pipeline(
    name='vertex-pipelines-sample',
    description='Vertex Piplines sample',
    pipeline_root='gs://vertex-pipelines-sample'
)
def pipeline(learning_rate: float = 0.1,
             max_depth: int = 10) -> None:
    preprocess_op = components.load_component_from_file(
        'components/preprocess/component.yaml'
    )    
    preprocess_task = preprocess_op(
        src_path='gs://workflow-example-dataset/penguins.csv',
        n_splits=3
    )

    train_op = components.load_component_from_file(
        'components/train/component.yaml'
    )    
    train_task = train_op(
        train_path=preprocess_task.outputs['train_path'],
        val_path=preprocess_task.outputs['val_path'],
        learning_rate=learning_rate,
        max_depth=max_depth,
    )
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
        'components/evaluate/component.yaml'
    )
    _ = evaluate_op(
        val_path=preprocess_task.outputs['val_path'],
        model_path=train_task.outputs['model_path'],
    )

    
compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path='vertex-pipelines-sample.json'
)

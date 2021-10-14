import json
import joblib
import argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score


def run(dataset_uri: str, artifact_uri: str, metrics_uri: str) -> None:
    df_val = pd.read_csv(f'{dataset_uri}/val.csv')
    x_val, y_val = df_val.drop(['target'], axis=1), df_val['target']

    model = joblib.load(f'{artifact_uri}/model.joblib')
    print(f'Model loaded from: {artifact_uri}')

    y_prob = model.predict(x_val)
    y_pred = y_prob.argmax(axis=-1)
    acc = accuracy_score(y_val, y_pred)
    print(f'Validation accuracy: {acc}')

    metrics_dir = Path(metrics_uri)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'metrics': [{
            'name': 'accuracy',
            'numberValue': acc,
            'format': 'PERCENTAGE'
        }]
    }
    with open(metrics_dir / 'mlpipeline_metrics.json', 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--dataset-uri', type=str)
    parser.add_argument('--artifact-uri', type=str)
    parser.add_argument('--metrics-uri', type=str)

    args = parser.parse_args()
    run(**vars(args))

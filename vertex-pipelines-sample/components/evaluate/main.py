import json
import argparse
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import accuracy_score


def run(val_path: str, model_path: str, metrics_path: str) -> None:
    df_val = pd.read_csv(val_path)
    x_val, y_val = df_val.drop(['target'], axis=1), df_val['target']

    model = lgb.Booster(model_file=model_path)
    print(f'Model loaded from: {model_path}')

    y_prob = model.predict(x_val)
    y_pred = y_prob.argmax(axis=-1)
    acc = accuracy_score(y_val, y_pred)
    print(f'Validation accuracy: {acc}')

    pardir = Path(metrics_path).parent
    pardir.mkdir(exist_ok=True, parents=True)

    metrics = {
        'metrics': [{
            'name': 'accuracy',
            'numberValue': acc,
            'format': 'PERCENTAGE'
        }]
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--val-path',
                        type=str,
                        required=True,
                        help='Path to the val csv')
    parser.add_argument('--model-path',
                        type=str,
                        required=True,
                        help='Path to the target model')
    parser.add_argument('--metrics-path',
                        type=str,
                        required=True,
                        help='Output metric path')

    args = parser.parse_args()
    run(**vars(args))

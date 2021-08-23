import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import accuracy_score


def plot_importance(features: List[str],
                    importances: np.ndarray,
                    savepath: str) -> None:
    df = pd.DataFrame({'feature': features, 'importance': importances})
    df = df.sort_values('importance', ascending=True)
    ax = df.plot.barh(y='importance', x='feature')
    ax.set_title('LightGBM importance')
    fig = ax.get_figure()
    fig.savefig(savepath, format='png')
    plt.close()


def run(val_path: str,
        model_path: str,
        metrics_path: str,
        visualize_path: str) -> None:
    df_val = pd.read_csv(val_path)
    x_val, y_val = df_val.drop(['target'], axis=1), df_val['target']

    model = lgb.Booster(model_file=model_path)
    print(f'Model loaded from: {model_path}')

    y_prob = model.predict(x_val)
    y_pred = y_prob.argmax(axis=-1)
    acc = accuracy_score(y_val, y_pred)
    print(f'Validation accuracy: {acc}')

    metrics = {'metrics': [
        {'name': 'accuracy', 'value': acc, 'format': 'RAW'},
    ]}

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    plot_importance(
        features=x_val.columns.to_list(),
        importances=model.feature_importance(),
        savepath=visualize_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--val-path', type=str, required=True,
                        help='Path to the val csv')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the target model')
    parser.add_argument('--mlpipeline-metrics', type=str, required=True,
                        help='Output metric path')
    parser.add_argument('--visualize-path', type=str, required=True,
                        help='Output visualization path')

    args = parser.parse_args()
    run(**vars(args))

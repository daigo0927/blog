import argparse
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import accuracy_score


SEED = 42


def run(train_path: str,
        val_path: str,
        model_path: str,
        learning_rate: float,
        max_depth: int,
        bagging_fraction: float,
        feature_fraction: float,
        lambda_l1: float,
        lambda_l2: float,
        min_data_in_leaf: int,
        num_leaves: int) -> None:
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    print(f'Data size: train: {df_train.shape}, val: {df_val.shape}')

    x_train, y_train = df_train.drop(['target'], axis=1), df_train['target']
    x_val, y_val = df_val.drop(['target'], axis=1), df_val['target']

    ds_train = lgb.Dataset(x_train, label=y_train)
    ds_val = lgb.Dataset(x_val, label=y_val)

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'bagging_fraction': bagging_fraction,
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'min_data_in_leaf': min_data_in_leaf,
        'num_leaves': num_leaves,
        'random_state': SEED,
        'verbose': -1
    }

    model = lgb.train(params,
                      ds_train,
                      num_boost_round=1000,
                      early_stopping_rounds=10,
                      valid_sets=[ds_train, ds_val],
                      verbose_eval=50)

    y_pred = model.predict(x_val, num_iteration=model.best_iteration)
    y_pred = y_pred.argmax(axis=-1)
    acc_val = accuracy_score(y_val, y_pred)
    print(f'Validation accuracy: {acc_val}')

    pardir = Path(model_path).parent
    pardir.mkdir(exist_ok=True, parents=True)
    model.save_model(model_path)
    print(f'Save model to: {model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--train-path', type=str, required=True,
                        help='Path to the train csv')
    parser.add_argument('--val-path', type=str, required=True,
                        help='Path to the val csv')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path for putting the trained model')
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--bagging-fraction', type=float, default=0.7)
    parser.add_argument('--feature-fraction', type=float, default=0.7)
    parser.add_argument('--lambda_l1', type=float, default=1.0)
    parser.add_argument('--lambda_l2', type=float, default=1.0)
    parser.add_argument('--min-data-in-leaf', type=int, default=10)
    parser.add_argument('--num-leaves', type=int, default=40)

    args = parser.parse_args()
    run(**vars(args))

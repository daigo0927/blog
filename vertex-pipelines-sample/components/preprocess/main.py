import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


SEED = 42


def run(src_path: str,
        n_splits: int,
        train_path: str,
        val_path: str) -> None:
    df = pd.read_csv(src_path)
    print(f"Load CSV from: {src_path}")

    df['target'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
    df = df.drop(['species', 'island', 'sex'], axis=1)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    idx_train, idx_val = next(cv.split(df, df['target']))
    df_train = df.iloc[idx_train]
    df_val = df.iloc[idx_val]

    pardir = Path(train_path).parent
    pardir.mkdir(exist_ok=True, parents=True)
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    print(f'Save train data to: {train_path}')
    print(f'Save val data to: {val_path}')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--src-path', type=str, required=True,
                        help='Source CSV file location')
    parser.add_argument('--n-splits', type=int, default=3,
                        help='Number of train/val splits. [3] as default')    
    parser.add_argument('--train-path', type=str, required=True,
                        help='Path to the output train csv')
    parser.add_argument('--val-path', type=str, required=True,
                        help='Path to the output val csv')
    args = parser.parse_args()

    run(**vars(args))

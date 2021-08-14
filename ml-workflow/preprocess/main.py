import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SEED = 42


def run(csv_path: str, output_dir: str, n_splits: int) -> None:
    df = pd.read_csv(csv_path)
    print(f"Load CSV from: {csv_path}")

    df['target'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
    df = df.drop(['species'], axis=1)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    idx_train, idx_val = next(cv.split(df, df['target']))
    df_train = df.iloc[idx_train]
    df_val = df.iloc[idx_val]

    df_train.to_csv(f'{output_dir}/train.csv', index=False)
    df_val.to_csv(f'{output_dir}/val.csv', index=False)
    print(f'Save train/val data to: {output_dir}')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--csv-path', type=str, required=True,
                        help='Source CSV file location')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='GCS location for writing outputs')    
    parser.add_argument('--n-splits', type=int, default=3,
                        help='Number of train/val splits. [3] as default')
    args = parser.parse_args()

    run(**vars(args))

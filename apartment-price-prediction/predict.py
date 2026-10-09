import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_csv', type=Path)
parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
def main():
    args = parser.parse_args()
    frame = pd.read_csv(args.input_csv, keep_default_na=False, na_values=[''])
    from housing.pipeline import prepare_features
    frame = prepare_features(frame)
    model = joblib.load(ROOT / 'models/best_model.joblib')
    result = pd.DataFrame({'prediction': model.predict(frame)})

    result.to_csv(args.output, index=False)
if __name__ == '__main__':
    main()

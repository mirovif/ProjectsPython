import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCLUDED = ['Order', 'PID', 'SalePrice', 'Mo Sold', 'Yr Sold', 'Sale Type', 'Sale Condition']

def load_frame(path):

    frame = pd.read_csv(path, sep='\t', keep_default_na=False, na_values=[''])
    if not frame['PID'].is_unique or frame['SalePrice'].isna().any() or (frame['SalePrice'] <= 0).any():
        raise ValueError('Expected one observation per property and positive, known sale prices.')
    return frame

def prepare_features(frame):
    features = frame.drop(columns=EXCLUDED, errors='ignore').copy()
    if 'MS SubClass' in features:
        features['MS SubClass'] = features['MS SubClass'].map(
            lambda value: str(int(value)) if pd.notna(value) else np.nan)
    for column in features.select_dtypes(include='object'):
        features[column] = features[column].map(
            lambda value: value.strip() if isinstance(value, str) else value)
    return features

def make_pipeline(model):
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=5)),
    ])
    preprocessing = ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('categorical', categorical, make_column_selector(dtype_include=object)),
    ])
    return Pipeline([('preprocess', preprocessing), ('model', model)])

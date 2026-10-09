import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_FEATURES = ['pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
NUMERIC = ['age', 'fare', 'family_size', 'fare_per_person', 'is_alone']
CATEGORICAL = ['sex', 'pclass', 'embarked', 'title']

class PassengerFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        result = X[RAW_FEATURES].copy()
        result['family_size'] = result['sibsp'] + result['parch'] + 1
        result['is_alone'] = (result['family_size'] == 1).astype(int)
        result['fare_per_person'] = result['fare'] / result['family_size']
        titles = result['name'].str.extract(r',\s*([^.]*)\.', expand=False).str.strip()
        result['title'] = titles.where(titles.isin(['Mr', 'Mrs', 'Miss', 'Master']), 'Other')
        result['pclass'] = result['pclass'].astype(str)
        return result[NUMERIC + CATEGORICAL]

def make_pipeline(model):
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ('numeric', numeric, NUMERIC), ('categorical', categorical, CATEGORICAL),
    ])
    return Pipeline([
        ('features', PassengerFeatures()), ('preprocess', preprocessing), ('model', model),
    ])

def ticket_groups(frame):

    values = frame['ticket'].fillna('').astype(str).str.upper().str.replace(r'\s+', '', regex=True)
    return np.array([value if value else f'unknown_{i}' for i, value in enumerate(values)])

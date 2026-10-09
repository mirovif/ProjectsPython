import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from housing.pipeline import make_pipeline, prepare_features

def test_preprocessing_never_learns_holdout_median_and_handles_unseen_category():
    train = pd.DataFrame({'area': [10., 20., np.nan], 'district': ['A', 'B', 'A']})
    holdout = pd.DataFrame({'area': [10000., np.nan], 'district': ['UNSEEN', 'A']})
    model = make_pipeline(Ridge()).fit(train, [1., 2., 1.5])
    imputer = model['preprocess'].named_transformers_['numeric']['impute']
    np.testing.assert_allclose(imputer.statistics_, [15.])
    assert np.isfinite(model.predict(holdout)).all()
    np.testing.assert_allclose(imputer.statistics_, [15.])

def test_identifiers_target_and_post_sale_information_are_removed():
    frame = pd.DataFrame({'Order': [1], 'PID': [99], 'SalePrice': [10], 'Sale Type': ['WD'],
                          'Sale Condition': ['Normal'], 'Yr Sold': [2010], 'Mo Sold': [1],
                          'MS SubClass': [20], 'Gr Liv Area': [1000]})
    features = prepare_features(frame)
    assert set(features.columns) == {'MS SubClass', 'Gr Liv Area'}
    assert features['MS SubClass'].iloc[0] == '20'

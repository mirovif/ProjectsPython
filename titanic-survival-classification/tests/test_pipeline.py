import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from titanic.pipeline import PassengerFeatures, make_pipeline, ticket_groups

def sample():
    return pd.DataFrame({'pclass': [1, 2, 3, 1, 2, 3],
                         'name': ['A, Mr. One', 'B, Mrs. Two', 'C, Miss. Three', 'D, Master. Four', 'E, Dr. Five', 'F, Mr. Six'],
                         'sex': ['male', 'female', 'female', 'male', 'male', 'male'],
                         'age': [20., np.nan, 30., 8., 50., 25.], 'sibsp': [0, 1, 0, 1, 0, 0],
                         'parch': [0, 0, 1, 1, 0, 0], 'fare': [10., 20., 30., 40., 50., 60.],
                         'embarked': ['S', 'C', 'S', 'Q', 'S', 'C']})

def test_features_are_stateless_and_ignore_leaked_labels():
    frame = sample().assign(survived=[0, 1, 1, 1, 0, 0], alive=['no', 'yes', 'yes', 'yes', 'no', 'no'])
    original = frame.copy(deep=True)
    features = PassengerFeatures().fit_transform(frame)
    assert not {'survived', 'alive', 'name', 'ticket'} & set(features.columns)
    assert features['family_size'].tolist() == [1, 2, 2, 3, 1, 1]
    assert features['title'].tolist() == ['Mr', 'Mrs', 'Miss', 'Master', 'Other', 'Mr']
    pd.testing.assert_frame_equal(frame, original)

def test_unseen_categories_missing_age_and_fold_local_median():
    model = make_pipeline(LogisticRegression(max_iter=1000)).fit(sample(), [0, 1, 1, 1, 0, 0])
    statistics = model['preprocess'].named_transformers_['numeric']['impute'].statistics_.copy()
    holdout = sample().iloc[:2].copy()
    holdout['age'] = [1000., np.nan]; holdout['embarked'] = ['UNKNOWN', 'C']
    probability = model.predict_proba(holdout)
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1.)
    np.testing.assert_allclose(model['preprocess'].named_transformers_['numeric']['impute'].statistics_, statistics)

def test_shared_tickets_stay_in_same_fold():
    frame = pd.DataFrame({'ticket': ['a 1', 'A1', 'B2', 'B2', 'C3', 'C3', 'D4', 'D4']})
    groups = ticket_groups(frame)
    assert groups[0] == groups[1]
    for train, test in StratifiedGroupKFold(n_splits=2).split(frame, [0, 1] * 4, groups):
        assert set(groups[train]).isdisjoint(groups[test])

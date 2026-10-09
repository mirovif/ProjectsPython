import json
import platform
import time
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from .data import ROOT, download_data
from .pipeline import RAW_FEATURES, make_pipeline, ticket_groups

SEED = 42
COLOR = '#256b85'

def save_figure(name):
    plt.tight_layout()
    plt.savefig(ROOT / 'images' / f'{name}.png', dpi=170, bbox_inches='tight')
    plt.close()

def main():
    start = time.perf_counter()
    for folder in ['images', 'reports', 'models']:
        (ROOT / folder).mkdir(exist_ok=True)
    plt.rcParams.update({'figure.figsize': (8, 4.8), 'axes.spines.top': False,
                        'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18,
                        'font.size': 11, 'axes.titlepad': 14})
    frame = pd.read_csv(download_data())
    if frame['survived'].isna().any() or not set(frame['survived'].unique()) <= {0, 1}:
        raise ValueError('Expected known binary survival labels.')
    X, y, groups = frame[RAW_FEATURES], frame['survived'], ticket_groups(frame)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(outer.split(X, y, groups))
    assert set(groups[train_idx]).isdisjoint(groups[test_idx])
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups[train_idx]
    pd.DataFrame({'row_index': X.index, 'split': np.where(X.index.isin(X_test.index), 'test', 'train')}).to_csv(
        ROOT / 'reports/split.csv', index=False)
    X_train.isna().sum().sort_values(ascending=False).to_csv(ROOT / 'reports/missing_train.csv', header=['missing'])
    X_train.describe(include='all').to_csv(ROOT / 'reports/eda_train.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    y_train.value_counts().sort_index().plot.bar(ax=axes[0], color=[COLOR, '#a4bbc3'], rot=0)
    axes[0].set(title='Survival · training data', xlabel='0 = died / 1 = survived', ylabel='Passengers')
    frame.iloc[train_idx].groupby('sex')['survived'].mean().plot.bar(ax=axes[1], color=COLOR, rot=0)
    axes[1].set(title='Survival by sex', ylabel='Survival rate', ylim=(0, 1), xlabel='Recorded sex')
    frame.iloc[train_idx].groupby('pclass')['survived'].mean().plot.bar(ax=axes[2], color=COLOR, rot=0)
    axes[2].set(title='Survival by ticket class', ylabel='Survival rate', ylim=(0, 1), xlabel='Ticket class')
    save_figure('eda')
    missing = X_train.isna().mean().sort_values(ascending=True)
    missing.plot.barh(color=COLOR)
    plt.title('Missing values · training data'); plt.xlabel('Fraction missing')
    save_figure('missing_values')
    specs = {
        'Majority baseline': (DummyClassifier(strategy='most_frequent'), {}),
        'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=SEED), {'model__C': [.1, 1., 10.]}),
        'Decision Tree': (DecisionTreeClassifier(random_state=SEED),
                          {'model__max_depth': [3, 5, 8], 'model__min_samples_leaf': [5, 15]}),
        'Random Forest': (RandomForestClassifier(n_estimators=160, random_state=SEED, n_jobs=1),
                          {'model__max_depth': [5, None], 'model__min_samples_leaf': [2, 5]}),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=120, random_state=SEED),
                              {'model__max_depth': [1, 2], 'model__learning_rate': [.05, .1]}),
    }
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    searches, rows = {}, []
    for name, (model, grid) in specs.items():
        search = GridSearchCV(make_pipeline(model), grid, scoring='roc_auc', cv=cv, n_jobs=1,
                              refit=True, error_score='raise')
        search.fit(X_train, y_train, groups=train_groups)
        searches[name] = search
        rows.append({'model': name, 'cv_roc_auc': float(search.best_score_),
                     'cv_roc_auc_std': float(search.cv_results_['std_test_score'][search.best_index_])})
        pd.DataFrame(search.cv_results_).to_csv(ROOT / 'reports' / (name.lower().replace(' ', '_') + '_cv.csv'), index=False)
        print(name, 'CV ROC-AUC', round(search.best_score_, 4), flush=True)
    winner = max(rows, key=lambda row: row['cv_roc_auc'])['model']
    for row in rows:
        model = searches[row['model']]
        predicted = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        row.update(accuracy=float(accuracy_score(y_test, predicted)),
                   precision=float(precision_score(y_test, predicted, zero_division=0)),
                   recall=float(recall_score(y_test, predicted, zero_division=0)),
                   f1=float(f1_score(y_test, predicted, zero_division=0)),
                   roc_auc=float(roc_auc_score(y_test, probability)))
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / 'reports/model_comparison.csv', index=False)
    best = searches[winner].best_estimator_
    predicted, probability = best.predict(X_test), best.predict_proba(X_test)[:, 1]

    ConfusionMatrixDisplay.from_predictions(y_test, predicted, display_labels=['Died', 'Survived'], cmap='Blues', colorbar=False)
    plt.title(f'{winner} · holdout · threshold 0.5'); plt.grid(False)
    save_figure('confusion_matrix')
    RocCurveDisplay.from_predictions(y_test, probability, name=winner, color=COLOR)
    plt.plot([0, 1], [0, 1], linestyle='--', color='#a4bbc3')
    plt.title('ROC curve · holdout'); save_figure('roc_curve')
    results.plot.barh(x='model', y=['cv_roc_auc', 'roc_auc'], color=[COLOR, '#a4bbc3'], figsize=(9, 4.8))
    plt.xlabel('ROC-AUC · higher is better'); plt.xlim(0, 1); plt.ylabel('')
    plt.title('Model comparison · selection uses group CV only'); plt.legend(['3-fold group CV', 'Group holdout'])
    save_figure('model_comparison')

    importance = permutation_importance(best, X_test, y_test, scoring='roc_auc',
                                        n_repeats=5, random_state=SEED, n_jobs=1)
    importance_frame = pd.DataFrame({'feature': X.columns, 'decrease_in_auc': importance.importances_mean,
                                    'std': importance.importances_std}).sort_values('decrease_in_auc', ascending=False)
    importance_frame.to_csv(ROOT / 'reports/permutation_importance.csv', index=False)
    top = importance_frame.iloc[::-1]
    plt.barh(top['feature'].replace({'name': 'Title (from name)'}), top['decrease_in_auc'], xerr=top['std'], color=COLOR)
    plt.xlabel('Decrease in holdout ROC-AUC after shuffling'); plt.title('Permutation importance · 5 repeats')
    save_figure('feature_importance')
    errors = X_test.drop(columns='name').copy()
    errors.insert(0, 'row_index', errors.index)
    errors['actual'] = y_test; errors['predicted'] = predicted; errors['probability'] = probability
    errors['incorrect'] = (y_test != predicted).astype(int)
    errors.to_csv(ROOT / 'reports/test_predictions.csv', index=False)
    errors[errors['incorrect'] == 1].to_csv(ROOT / 'reports/misclassified.csv', index=False)
    segments = errors.groupby('sex')['incorrect'].agg(['count', 'mean'])
    segments.to_csv(ROOT / 'reports/error_by_sex.csv')
    joblib.dump(best, ROOT / 'models/best_model.joblib')
    summary = {'selected_model': winner, 'selection_metric': '3-fold ticket-group CV ROC-AUC on training only',
               'seed': SEED, 'rows': len(frame), 'train_rows': len(X_train), 'test_rows': len(X_test),
               'train_survival_rate': float(y_train.mean()), 'test_survival_rate': float(y_test.mean()),
               'overlapping_tickets': len(set(groups[train_idx]) & set(groups[test_idx])),
               'best_parameters': searches[winner].best_params_, 'threshold': .5,
               'confusion_matrix': confusion_matrix(y_test, predicted).tolist(), 'results': rows,
               'error_by_sex': segments.reset_index().to_dict('records'),
               'python': platform.python_version(), 'sklearn': sklearn.__version__,
               'elapsed_seconds': round(time.perf_counter() - start, 2)}
    (ROOT / 'reports/summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('Selected:', winner, flush=True)

if __name__ == '__main__':
    main()

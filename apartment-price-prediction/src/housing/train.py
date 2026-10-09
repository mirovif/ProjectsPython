import json
import platform
import time
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from .data import ROOT, download_data
from .pipeline import load_frame, prepare_features, make_pipeline

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
    frame = load_frame(download_data())
    X, y = prepare_features(frame), frame['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)
    pd.DataFrame({'row_index': X.index, 'split': np.where(X.index.isin(X_test.index), 'test', 'train')}).to_csv(
        ROOT / 'reports/split.csv', index=False)

    X_train.isna().sum().sort_values(ascending=False).to_csv(ROOT / 'reports/missing_train.csv', header=['missing'])
    X_train.describe(include='all').to_csv(ROOT / 'reports/eda_train.csv')
    plt.hist(y_train, bins=35, color=COLOR, edgecolor='white')
    plt.title('Ames sale prices · training data'); plt.xlabel('Sale price (USD)'); plt.ylabel('Properties')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:,.0f}k'))
    save_figure('target_distribution')
    correlations = X_train.select_dtypes(include=np.number).corrwith(y_train).dropna()
    correlations.reindex(correlations.abs().nlargest(12).index).sort_values().plot.barh(color=COLOR)
    plt.title('Strongest numeric correlations · training data'); plt.xlabel('Pearson correlation with sale price')
    save_figure('feature_analysis')
    plt.scatter(X_train['Gr Liv Area'], y_train, alpha=.35, s=16, color=COLOR)
    plt.title('Living area and sale price · training data'); plt.xlabel('Above-ground living area (sq ft)'); plt.ylabel('Sale price (USD)')
    save_figure('area_price')
    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    specs = {
        'Median baseline': (DummyRegressor(strategy='median'), {}),
        'Ridge regression': (Ridge(), {'model__alpha': [.1, 1., 10., 100.]}),
        'Decision Tree': (DecisionTreeRegressor(random_state=SEED),
                          {'model__max_depth': [4, 8, None], 'model__min_samples_leaf': [5, 15]}),
        'Random Forest': (RandomForestRegressor(n_estimators=160, random_state=SEED, n_jobs=1),
                          {'model__max_depth': [12, None], 'model__min_samples_leaf': [2, 5]}),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=200, random_state=SEED),
                              {'model__max_depth': [2, 3], 'model__learning_rate': [.05, .1]}),
    }
    searches, rows = {}, []
    for name, (model, grid) in specs.items():
        search = GridSearchCV(make_pipeline(model), grid, scoring='neg_root_mean_squared_error',
                              cv=cv, n_jobs=1, refit=True, error_score='raise')
        search.fit(X_train, y_train)
        searches[name] = search
        rows.append({'model': name, 'cv_rmse': -float(search.best_score_),
                     'cv_rmse_std': float(search.cv_results_['std_test_score'][search.best_index_])})
        pd.DataFrame(search.cv_results_).to_csv(ROOT / 'reports' / (name.lower().replace(' ', '_') + '_cv.csv'), index=False)
        print(name, 'CV RMSE', round(-search.best_score_, 2), flush=True)
    winner = min(rows, key=lambda row: row['cv_rmse'])['model']

    for row in rows:
        prediction = searches[row['model']].predict(X_test)
        mse = mean_squared_error(y_test, prediction)
        row.update(mae=float(mean_absolute_error(y_test, prediction)), mse=float(mse),
                   rmse=float(np.sqrt(mse)), r2=float(r2_score(y_test, prediction)))
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / 'reports/model_comparison.csv', index=False)
    best = searches[winner].best_estimator_
    prediction = best.predict(X_test)
    errors = X_test.copy()
    errors.insert(0, 'row_index', errors.index)
    errors['actual'] = y_test; errors['predicted'] = prediction
    errors['residual'] = y_test - prediction; errors['absolute_error'] = abs(errors['residual'])
    errors.to_csv(ROOT / 'reports/test_predictions.csv', index=False)
    errors.nlargest(10, 'absolute_error').to_csv(ROOT / 'reports/worst_errors.csv', index=False)
    bounds = np.r_[-np.inf, y_train.quantile([.25, .5, .75]).values, np.inf]
    errors['price_band'] = pd.cut(y_test, bounds, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    segments = errors.groupby('price_band', observed=True)['absolute_error'].agg(['count', 'mean'])
    segments.to_csv(ROOT / 'reports/error_by_price_band.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_test, prediction, alpha=.55, s=18, color=COLOR)
    limits = [min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())]
    axes[0].plot(limits, limits, color='#bf624c', linestyle='--')
    axes[0].set(xlabel='Actual (USD)', ylabel='Predicted (USD)', title=f'{winner} · holdout')
    axes[1].scatter(prediction, y_test - prediction, alpha=.55, s=18, color=COLOR)
    axes[1].axhline(0, color='#bf624c', linestyle='--')
    axes[1].set(xlabel='Predicted (USD)', ylabel='Actual − predicted (USD)', title='Residuals · holdout')
    save_figure('predictions_and_errors')
    results.plot.barh(x='model', y=['cv_rmse', 'rmse'], color=[COLOR, '#a4bbc3'], figsize=(9, 4.8))
    plt.xlabel('RMSE (USD) · lower is better'); plt.ylabel(''); plt.title('Model comparison · selection uses CV only')
    plt.legend(['3-fold CV', 'Holdout'])
    save_figure('model_comparison')
    importance = permutation_importance(best, X_test, y_test, scoring='neg_root_mean_squared_error',
                                        n_repeats=5, random_state=SEED, n_jobs=1)
    importance_frame = pd.DataFrame({'feature': X.columns, 'increase_in_rmse': importance.importances_mean,
                                    'std': importance.importances_std}).sort_values('increase_in_rmse', ascending=False)
    importance_frame.to_csv(ROOT / 'reports/permutation_importance.csv', index=False)
    top = importance_frame.head(12).iloc[::-1]
    plt.barh(top['feature'], top['increase_in_rmse'], xerr=top['std'], color=COLOR)
    plt.xlabel('Increase in holdout RMSE after shuffling (USD)'); plt.title('Permutation importance · 5 repeats')
    save_figure('feature_importance')
    joblib.dump(best, ROOT / 'models/best_model.joblib')
    summary = {'selected_model': winner, 'selection_metric': '3-fold CV RMSE on training only',
               'seed': SEED, 'rows': len(frame), 'features': X.shape[1], 'train_rows': len(X_train),
               'test_rows': len(X_test), 'best_parameters': searches[winner].best_params_,
               'results': rows, 'price_band_errors': segments.reset_index().to_dict('records'),
               'python': platform.python_version(), 'sklearn': sklearn.__version__,
               'elapsed_seconds': round(time.perf_counter() - start, 2)}
    (ROOT / 'reports/summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('Selected:', winner, flush=True)

if __name__ == '__main__':
    main()

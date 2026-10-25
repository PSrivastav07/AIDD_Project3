"""Model registry for the classical (non-graph) learners.

All models are scikit-learn compatible estimators (``fit`` / ``predict``). Descriptor matrices may
contain NaN: every pipeline that cannot handle NaN starts with a median imputer.

Names
-----
``mean``         predicts the training mean (sanity floor)
``esol_linear``  ordinary least squares - intended for the ``esol4`` features (refit of Delaney's ESOL)
``ridge``        RidgeCV on standardised features
``rf``           random forest
``lgbm``         LightGBM gradient boosting
``knn``          k-nearest neighbours with Jaccard/Tanimoto distance (binary fingerprints)
``mlp``          multilayer perceptron on standardised features
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DEFAULT_PARAMS = {
    "rf": {"n_estimators": 500, "max_features": 0.3, "min_samples_leaf": 1},
    "lgbm": {"n_estimators": 2000, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 10,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.3, "reg_lambda": 1.0},
    "knn": {"n_neighbors": 5, "weights": "distance"},
    "mlp": {"hidden_layer_sizes": (512, 256), "alpha": 1e-3, "learning_rate_init": 1e-3,
            "early_stopping": True, "max_iter": 500},
}


def _binarize(X):
    return np.asarray(X) > 0


def get_model(name: str, seed: int = 0, n_jobs: int = 4, params: dict | None = None):
    """Build an unfitted estimator. ``params`` override the defaults of the final estimator."""
    p = {**DEFAULT_PARAMS.get(name, {}), **(params or {})}
    impute = SimpleImputer(strategy="median", keep_empty_features=True)
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "esol_linear":
        return make_pipeline(impute, LinearRegression())
    if name == "ridge":
        return make_pipeline(impute, VarianceThreshold(0.0), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 4, 30)))
    if name == "rf":
        return make_pipeline(impute, RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **p))
    if name == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1, **p)
    if name == "knn":
        return Pipeline([("bin", FunctionTransformer(_binarize)),
                         ("knn", KNeighborsRegressor(metric="jaccard", algorithm="brute", n_jobs=n_jobs, **p))])
    if name == "mlp":
        return make_pipeline(impute, VarianceThreshold(0.0), StandardScaler(),
                             MLPRegressor(random_state=seed, **p))
    raise ValueError(f"unknown model {name!r}")


def fit_model(model, X_train, y_train, X_val=None, y_val=None):
    """Fit; LightGBM uses the validation set for early stopping when it is given."""
    name = type(model).__name__
    if name == "LGBMRegressor" and X_val is not None:
        import inspect

        import lightgbm as lgb

        cb = [lgb.early_stopping(100, verbose=False)]
        if "eval_X" in inspect.signature(model.fit).parameters:  # lightgbm >= 4.6
            model.fit(X_train, y_train, eval_X=(X_val,), eval_y=(y_val,), eval_metric="l1", callbacks=cb)
        else:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l1", callbacks=cb)
    else:
        model.fit(X_train, y_train)
    return model


def predict_with_std(model, X) -> tuple[np.ndarray, np.ndarray | None]:
    """Prediction plus a per-sample spread when the model provides one (RF: across-tree std)."""
    est = model[-1] if isinstance(model, Pipeline) else model
    if isinstance(est, RandomForestRegressor):
        Xt = model[:-1].transform(X) if isinstance(model, Pipeline) else X
        per_tree = np.stack([t.predict(Xt) for t in est.estimators_])
        return per_tree.mean(0), per_tree.std(0)
    return model.predict(X), None

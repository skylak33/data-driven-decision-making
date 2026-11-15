"""Учебный градиентный бустинг: реализация без готовых ансамблей."""
from __future__ import annotations

from collections import defaultdict
from inspect import signature

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm


def _category_column(values):
    # Единое представление пропущенных и ранее неизвестных категорий.
    return pd.Series(values, dtype="string").fillna("<NA>")


class TargetEncoder(BaseEstimator):
    """Счётчики; fit_transform использует только предыдущие строки при ordered=True."""

    def __init__(self, cat_features=(), ordered=True, smoothing=1.0, prior=0.5):
        self.cat_features = cat_features
        self.ordered = ordered
        self.smoothing = smoothing
        self.prior = prior

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y, dtype=float)
        self.maps_ = {}
        for j in (() if self.cat_features is None else self.cat_features):
            col = _category_column(X[:, j])
            stats = pd.DataFrame({"category": col, "target": y}).groupby("category")["target"].agg(["sum", "count"])
            # Полные статистики применяются только к новым объектам.
            self.maps_[j] = (stats["sum"] / stats["count"]).to_dict()
        return self

    def transform(self, X):
        check_is_fitted(self, "maps_")
        result = np.asarray(X, dtype=object).copy()
        for j, mapping in self.maps_.items():
            result[:, j] = _category_column(result[:, j]).map(mapping).fillna(self.prior).to_numpy()
        return result.astype(float)

    def fit_transform(self, X, y):
        self.fit(X, y)
        if not self.ordered:
            return self.transform(X)
        result = np.asarray(X, dtype=object).copy()
        y = np.asarray(y, dtype=float)
        for j in (() if self.cat_features is None else self.cat_features):
            col = _category_column(result[:, j])
            frame = pd.DataFrame({"category": col, "target": y})
            group = frame.groupby("category", sort=False)["target"]
            preceding_sum = group.cumsum().to_numpy() - y
            preceding_count = frame.groupby("category", sort=False).cumcount().to_numpy()
            result[:, j] = (preceding_sum + self.smoothing * self.prior) / (preceding_count + self.smoothing)
        return result.astype(float)


class Quantizer(BaseEstimator):
    """Пороги оцениваются только на train; NaN выделен отдельный бин 0."""

    def __init__(self, quantization_type="quantile", nbins=255, features=None, random_state=None):
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.features = features
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.nbins < 2:
            raise ValueError("nbins must be >= 2")
        self.borders_ = {}
        features = range(X.shape[1]) if self.features is None else self.features
        for j in features:
            col = X[:, j]
            finite = np.isfinite(col)
            values = col[finite]
            if not len(values) or np.unique(values).size < 2:
                borders = np.array([])
            elif self.quantization_type == "uniform":
                borders = np.linspace(values.min(), values.max(), self.nbins + 1)[1:-1]
            elif self.quantization_type == "quantile":
                borders = np.unique(np.quantile(values, np.arange(1, self.nbins) / self.nbins))
                borders = borders[(borders > values.min()) & (borders < values.max())]
            elif self.quantization_type == "piecewise":
                if y is None:
                    raise ValueError("piecewise quantization requires training targets")
                tree = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=self.nbins,
                                              min_samples_leaf=max(2, len(values) // 1000),
                                              random_state=self.random_state)
                tree.fit(values[:, None], np.asarray(y)[finite])
                borders = np.sort(tree.tree_.threshold[tree.tree_.feature >= 0])
            else:
                raise ValueError("Unknown quantization_type")
            self.borders_[j] = borders
        return self

    def transform(self, X):
        check_is_fitted(self, "borders_")
        result = np.asarray(X, dtype=float).copy()
        for j, borders in self.borders_.items():
            col = result[:, j]
            result[:, j] = np.where(np.isnan(col), 0, np.searchsorted(borders, col, side="right") + 1)
        return result


def find_best_split(feature_vector, grad_vector, hess_vector, l2=1.0, min_samples_leaf=1):
    """Все допустимые числовые пороги и gain; grad_vector — АНТИградиент."""
    l2 = 0.0 if l2 is None else l2
    x, g, h = map(np.asarray, (feature_vector, grad_vector, hess_vector))
    order = np.argsort(x, kind="stable")
    x, g, h = x[order], g[order], h[order]
    positions = np.flatnonzero(x[:-1] != x[1:])
    positions = positions[(positions + 1 >= min_samples_leaf) & (len(x) - positions - 1 >= min_samples_leaf)]
    if not len(positions):
        return np.array([]), np.array([]), None, None
    G, H = np.cumsum(g), np.cumsum(h)
    gl, hl = G[positions], H[positions]
    gains = gl**2 / (hl + l2) + (G[-1] - gl)**2 / (H[-1] - hl + l2) - G[-1]**2 / (H[-1] + l2)
    thresholds = x[positions] + (x[positions + 1] - x[positions]) / 2
    best = int(np.argmax(gains))
    return thresholds, gains, thresholds[best], gains[best]


class NewtonTreeRegressor(RegressorMixin, BaseEstimator):
    """Дерево с критерием G²/(H+λ) и нативными разбиениями категорий."""

    def __init__(self, max_depth=3, min_samples_leaf=20, l2=1.0, cat_features=(), random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.cat_features = cat_features
        self.random_state = random_state

    def fit(self, X, y, hess_vector=None, sample_weight=None):
        X, g = np.asarray(X), np.asarray(y, dtype=float)
        h = np.ones_like(g) if hess_vector is None else np.asarray(hess_vector, dtype=float)
        h = np.maximum(h, 1e-8)
        if sample_weight is not None:
            g, h = g * sample_weight, h * sample_weight
        self.l2_ = 0.0 if self.l2 is None else self.l2
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root_ = self._fit_node(X, g, h, 0)
        total = self.feature_importances_.sum()
        if total:
            self.feature_importances_ /= total
        return self

    def _fit_node(self, X, g, h, depth):
        # Именно отношение СУММ, а не среднее отношений из опечатки в условии.
        node = {"value": g.sum() / (h.sum() + self.l2_)}
        if depth >= self.max_depth or len(g) < 2 * self.min_samples_leaf:
            return node
        best_gain, best = 1e-12, None
        for j in range(X.shape[1]):
            categories = None
            if j in self.cat_features:
                codes, levels = pd.factorize(_category_column(X[:, j]), sort=True)
                G = np.bincount(codes, weights=g)
                H = np.bincount(codes, weights=h)
                category_order = np.argsort(G / (H + self.l2_))
                rank = np.empty(len(levels), dtype=int)
                rank[category_order] = np.arange(len(levels))
                col = rank[codes].astype(float)
                categories = np.asarray(levels)[category_order]
            else:
                col = X[:, j].astype(float)
            _, _, threshold, gain = find_best_split(col, g, h, self.l2_, self.min_samples_leaf)
            if gain is not None and gain > best_gain:
                best_gain = gain
                best = (j, threshold, col <= threshold, categories)
        if best is None:
            return node
        j, threshold, left, categories = best
        self.feature_importances_[j] += best_gain
        node.update(feature=j, threshold=threshold,
                    categories=None if categories is None else frozenset(categories[:int(np.floor(threshold)) + 1]),
                    left=self._fit_node(X[left], g[left], h[left], depth + 1),
                    right=self._fit_node(X[~left], g[~left], h[~left], depth + 1))
        return node

    def predict(self, X):
        check_is_fitted(self, "root_")
        X = np.asarray(X)
        out = np.empty(len(X))
        def walk(node, rows):
            if "feature" not in node:
                out[rows] = node["value"]
                return
            col = X[rows, node["feature"]]
            if node["categories"] is None:
                mask = col.astype(float) <= node["threshold"]
            else:
                mask = _category_column(col).isin(node["categories"]).to_numpy()
            walk(node["left"], rows[mask])
            walk(node["right"], rows[~mask])
        walk(self.root_, np.arange(len(X)))
        return out


class Boosting(ClassifierMixin, BaseEstimator):
    def __init__(self, base_model_class=DecisionTreeRegressor, base_model_params=None,
                 n_estimators=20, learning_rate=0.05, random_state=None, verbose=True,
                 early_stopping_rounds=0, eval_metric=None, cat_features=None,
                 ordered=True, subsample=1.0, bagging_temperature=1.0,
                 bootstrap_type="Bernoulli", rsm=1.0, goss=False, goss_k=0.2,
                 quantization_type=None, nbins=255, l2=0.0,
                 dart=False, dropout_rate=0.05, loss="BCE", focal_gamma=2.0):
        self.base_model_class = base_model_class
        self.base_model_params = base_model_params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.cat_features = cat_features
        self.ordered = ordered
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.goss = goss
        self.goss_k = goss_k
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.l2 = l2
        self.dart = dart
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.focal_gamma = focal_gamma

    @staticmethod
    def sigmoid(x):
        return expit(x)

    def loss_fn(self, y, z):
        margin = np.asarray(y) * np.asarray(z)
        bce = np.logaddexp(0, -margin)
        if self.loss.upper() == "BCE":
            return bce.mean()
        return (expit(-margin)**self.focal_gamma * bce).mean()

    def grad_fn(self, y, z):
        y = np.asarray(y)
        margin = y * z
        q = expit(-margin)
        if self.loss.upper() == "BCE" or self.focal_gamma == 0:
            return -y * q
        p = expit(margin)
        logp = -np.logaddexp(0, -margin)
        return y * q**self.focal_gamma * (self.focal_gamma * p * logp - q)

    def hess_fn(self, y, z):
        if self.loss.upper() != "BCE":
            raise ValueError("Newton trees are implemented for BCE only")
        p = expit(np.asarray(y) * z)
        return np.maximum(p * (1 - p), 1e-8)

    def _validate_params_values(self):
        if not 0 < self.learning_rate <= 1 or self.n_estimators < 1:
            raise ValueError("Require learning_rate in (0, 1] and n_estimators >= 1")
        if not 0 < self.subsample <= 1 or not 0 < self.rsm <= 1:
            raise ValueError("subsample and rsm must be in (0, 1]")
        if not 0 <= self.goss_k < 1 or not 0 <= self.dropout_rate <= 1:
            raise ValueError("Invalid goss_k or dropout_rate")
        if self.loss.upper() not in {"BCE", "FOCAL"} or self.focal_gamma < 0:
            raise ValueError("loss must be BCE or Focal; focal_gamma >= 0")
        if self.bootstrap_type not in {None, "Bernoulli", "Bayesian"}:
            raise ValueError("Unknown bootstrap_type")
        if self.bagging_temperature < 0 or (self.l2 or 0) < 0 or (self.early_stopping_rounds or 0) < 0:
            raise ValueError("Regularization parameters must be nonnegative")
        if self.l2 and "l2" not in signature(self.base_model_class).parameters:
            raise ValueError("l2 requires a base learner with Newton/l2 support")

    def _encode_y(self, y):
        y = np.asarray(y)
        if not np.isin(y, self.classes_).all():
            raise ValueError("Unknown target labels")
        return np.where(y == self.classes_[1], 1.0, -1.0)

    def _cat_fit(self, X, y):
        self.encoder_ = TargetEncoder(self.cat_features_, ordered=self.ordered)
        return self.encoder_.fit_transform(X, (y > 0).astype(float))

    def _cat_transform(self, X):
        return self.encoder_.transform(X)

    def _prepare_fit(self, X, y):
        self.native_categories_ = issubclass(self.base_model_class, NewtonTreeRegressor)
        if self.native_categories_:
            return np.asarray(X, dtype=object).copy()
        data = self._cat_fit(X, y)
        numeric = [j for j in range(data.shape[1]) if j not in self.cat_features_]
        self.quantizer_ = None
        if self.quantization_type is not None:
            self.quantizer_ = Quantizer(self.quantization_type, self.nbins, numeric, self.random_state).fit(data, y)
            data = self.quantizer_.transform(data)
        self.fill_values_ = np.nanmedian(np.where(np.isfinite(data), data, np.nan), axis=0)
        self.fill_values_ = np.nan_to_num(self.fill_values_)
        return np.where(np.isfinite(data), data, self.fill_values_)

    def _prepare_predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected X shape")
        if self.native_categories_:
            return X
        data = self._cat_transform(X)
        if self.quantizer_ is not None:
            data = self.quantizer_.transform(data)
        return np.where(np.isfinite(data), data, self.fill_values_)

    def _initialize(self, X, y):
        self._validate_params_values()
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(y) != len(X) or not len(y):
            raise ValueError("X must be 2D and y 1D, with matching nonempty rows")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Binary classification requires exactly two classes")
        self.n_features_in_ = X.shape[1]
        self.cat_features_ = () if self.cat_features is None else tuple(self.cat_features)
        self.models, self.gammas, self.features_ = [], [], []
        self.history = defaultdict(list)
        self.rng_ = np.random.default_rng(self.random_state)
        self._raw_X_train = X.copy()
        self._y_train = self._encode_y(y)
        self._X_train = self._prepare_fit(X, self._y_train)
        self._train_predictions = np.zeros(len(y))
        self._train_model_predictions = []
        self._valid_model_predictions = []
        self._X_valid = None
        self.n_estimators_ = 0

    def _sample(self, residual):
        n = len(residual)
        if self.goss:
            n_top = min(n - 1, max(1, int(n * self.goss_k))) if self.goss_k else 0
            # Случайный порядок устраняет систематический выбор при равных градиентах.
            order = self.rng_.permutation(n)
            order = order[np.argsort(-np.abs(residual[order]), kind="stable")]
            top, rest = order[:n_top], order[n_top:]
            mask = self.rng_.random(len(rest)) < self.subsample
            rows = np.r_[top, rest[mask]]
            weights = np.r_[np.ones(len(top)), np.full(mask.sum(), 1 / self.subsample)]
            if not len(rows):
                rows, weights = order[:1], np.ones(1)
            return rows, weights
        if self.bootstrap_type == "Bernoulli":
            rows = np.flatnonzero(self.rng_.random(n) < self.subsample)
            if not len(rows):
                rows = self.rng_.integers(n, size=1)
            return rows, None
        if self.bootstrap_type == "Bayesian":
            u = np.maximum(self.rng_.random(n), np.finfo(float).tiny)
            return np.arange(n), (-np.log(u))**self.bagging_temperature
        return np.arange(n), None

    def _boost_step(self):
        X, y = self._X_train, self._y_train
        old_predictions = self._train_predictions.copy()
        dropped = np.array([], dtype=int)
        if self.dart and self.models and self.dropout_rate > 0:
            dropped = np.flatnonzero(self.rng_.random(len(self.models)) < self.dropout_rate)
            if not len(dropped):
                dropped = self.rng_.integers(len(self.models), size=1)
            for j in dropped:
                old_predictions -= self.learning_rate * self.gammas[j] * self._train_model_predictions[j]
        residual = -self.grad_fn(y, old_predictions)
        rows, weights = self._sample(residual)
        cols = np.flatnonzero(self.rng_.random(X.shape[1]) < self.rsm)
        if not len(cols):
            cols = self.rng_.integers(X.shape[1], size=1)
        params = dict(self.base_model_params or {})
        supported = signature(self.base_model_class).parameters
        if "random_state" in supported:
            params.setdefault("random_state", int(self.rng_.integers(2**31 - 1)))
        if self.native_categories_:
            params["cat_features"] = [i for i, j in enumerate(cols) if j in self.cat_features_]
            params["l2"] = 0.0 if self.l2 is None else self.l2
        model = self.base_model_class(**params)
        fit_params = {}
        if weights is not None:
            fit_params["sample_weight"] = weights
        if self.native_categories_:
            fit_params["hess_vector"] = self.hess_fn(y, old_predictions)[rows]
        model.fit(X[np.ix_(rows, cols)], residual[rows], **fit_params)
        prediction = model.predict(X[:, cols])
        gamma = 1.0 if self.native_categories_ else self._find_optimal_gamma(y, old_predictions, prediction)
        if len(dropped):
            k = len(dropped)
            # 1/k для нового дерева, затем k/(k+1) для нового И удалённых.
            for j in dropped:
                old_gamma = self.gammas[j]
                self.gammas[j] *= k / (k + 1)
                self._train_predictions += self.learning_rate * (self.gammas[j] - old_gamma) * self._train_model_predictions[j]
                if self._X_valid is not None:
                    self._valid_predictions += self.learning_rate * (self.gammas[j] - old_gamma) * self._valid_model_predictions[j]
            gamma /= k + 1
        self.models.append(model)
        self.gammas.append(float(gamma))
        self.features_.append(cols)
        self._train_predictions += self.learning_rate * gamma * prediction
        if self.dart:
            self._train_model_predictions.append(prediction)
        if self._X_valid is not None:
            valid_prediction = model.predict(self._X_valid[:, cols])
            self._valid_predictions += self.learning_rate * gamma * valid_prediction
            if self.dart:
                self._valid_model_predictions.append(valid_prediction)
        self.n_estimators_ = len(self.models)

    def _record_history(self):
        for name, y, pred in [("train", self._y_train, self._train_predictions),
                              ("valid", getattr(self, "_y_valid", None), getattr(self, "_valid_predictions", None))]:
            if pred is not None:
                self.history[name + "_loss"].append(self.loss_fn(y, pred))
                self.history[name + "_roc_auc"].append(roc_auc_score(y > 0, pred))

    def partial_fit(self, X, y):
        """Добавить один алгоритм; статистики предобработки фиксируются первым вызовом."""
        if not hasattr(self, "classes_"):
            self._initialize(X, y)
        elif not (np.array_equal(np.asarray(X), self._raw_X_train, equal_nan=False)
                  and np.array_equal(self._encode_y(y), self._y_train)):
            raise ValueError("partial_fit expects the same training batch; use fit for new data")
        self.history = defaultdict(list, {k: list(v) for k, v in self.history.items()})
        self._boost_step()
        self._record_history()
        return self

    def fit(self, X_train, y_train, eval_set=None, use_best_model=False):
        self._initialize(X_train, y_train)
        if eval_set is not None:
            self._X_valid = self._prepare_predict(eval_set[0])
            self._y_valid = self._encode_y(eval_set[1])
            self._valid_predictions = np.zeros(len(self._y_valid))
        else:
            self._y_valid = self._valid_predictions = None
        if (use_best_model or self.early_stopping_rounds) and eval_set is None:
            raise ValueError("Early stopping and use_best_model require eval_set")
        metric = self.eval_metric or ("valid_loss" if eval_set is not None else "train_loss")
        if metric not in {"train_loss", "train_roc_auc", "valid_loss", "valid_roc_auc"}:
            raise ValueError("eval_metric must be a history key")
        if metric.startswith("valid") and eval_set is None:
            raise ValueError("Validation metric requires eval_set")
        maximize = metric.endswith("roc_auc")
        best_score = -np.inf if maximize else np.inf
        best_count, best_gammas, stale = 0, None, 0
        iterator = tqdm(range(self.n_estimators)) if self.verbose else range(self.n_estimators)
        for _ in iterator:
            self._boost_step()
            self._record_history()
            score = self.history[metric][-1]
            improved = score > best_score + 1e-12 if maximize else score < best_score - 1e-12
            if improved:
                best_score, best_count, stale = score, len(self.models), 0
                best_gammas = self.gammas.copy()
            else:
                stale += 1
            if self.early_stopping_rounds and stale >= self.early_stopping_rounds:
                break
        self.best_iteration_ = best_count - 1
        self.best_score_ = best_score
        self.stopped_iteration_ = len(self.models) - 1
        if use_best_model:
            self.models = self.models[:best_count]
            self.features_ = self.features_[:best_count]
            # В DART нельзя просто обрезать текущие веса: они менялись после лучшей итерации.
            self.gammas = best_gammas
            self._train_model_predictions = self._train_model_predictions[:best_count]
            self._valid_model_predictions = self._valid_model_predictions[:best_count]
            self._train_predictions = self._decision_prepared(self._X_train)
            if self._X_valid is not None:
                self._valid_predictions = self._decision_prepared(self._X_valid)
        self.n_estimators_ = len(self.models)
        self.history = {k: np.asarray(v) for k, v in self.history.items()}
        return self

    def _decision_prepared(self, X):
        result = np.zeros(len(X))
        for model, gamma, cols in zip(self.models, self.gammas, self.features_):
            result += self.learning_rate * gamma * model.predict(X[:, cols])
        return result

    def decision_function(self, X):
        check_is_fitted(self, "classes_")
        return self._decision_prepared(self._prepare_predict(X))

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack((1 - p, p))

    def predict(self, X):
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]

    def _find_optimal_gamma(self, y, old_predictions, new_predictions):
        # Для выпуклой BCE корень производной точнее и быстрее сетки из 100 точек.
        if self.loss.upper() == "BCE":
            def derivative(gamma):
                return np.mean(self.grad_fn(y, old_predictions + gamma * new_predictions) * new_predictions)
            if derivative(0) >= 0:
                return 0.0
            if derivative(1) <= 0:
                return 1.0
            return float(brentq(derivative, 0, 1))
        objective = lambda gamma: self.loss_fn(y, old_predictions + gamma * new_predictions)
        result = minimize_scalar(objective, bounds=(0, 1), method="bounded")
        candidates = [0.0, 1.0, float(result.x)]
        return min(candidates, key=objective)

    def score(self, X, y):
        return roc_auc_score(np.asarray(y) == self.classes_[1], self.predict_proba(X)[:, 1])

    def plot_history(self, keys):
        import matplotlib.pyplot as plt
        keys = [keys] if isinstance(keys, str) else list(keys)
        fig, ax = plt.subplots(figsize=(8, 4))
        for key in keys:
            ax.plot(np.arange(1, len(self.history[key]) + 1), self.history[key], label=key)
        ax.set(xlabel="Итерация", ylabel="Значение метрики")
        ax.legend()
        fig.tight_layout()
        return ax

    def get_feature_importance(self, X=None, y=None, type="split"):
        check_is_fitted(self, "classes_")
        importance = np.zeros(self.n_features_in_)
        if type == "split":
            for model, gamma, cols in zip(self.models, self.gammas, self.features_):
                importance[cols] += abs(gamma) * model.feature_importances_
        elif type == "gain":
            if X is None or y is None:
                raise ValueError("gain importance requires X and y")
            data, target = self._prepare_predict(X), self._encode_y(y)
            current = np.zeros(len(data))
            for model, gamma, cols in zip(self.models, self.gammas, self.features_):
                if not hasattr(model, "tree_"):
                    raise ValueError("gain importance requires sklearn regression trees")
                tree = model.tree_
                path = model.decision_path(data[:, cols]).tocsc()
                values = tree.value[:, 0, 0]
                scale = self.learning_rate * gamma
                for parent in np.flatnonzero(tree.feature >= 0):
                    feature = cols[tree.feature[parent]]
                    for child in (tree.children_left[parent], tree.children_right[parent]):
                        rows = path.indices[path.indptr[child]:path.indptr[child + 1]]
                        if not len(rows):
                            continue
                        before = current[rows] + scale * values[parent]
                        after = current[rows] + scale * values[child]
                        # Точная разность логистического loss на каждом ребре пути.
                        reduction = self.loss_fn(target[rows], before) - self.loss_fn(target[rows], after)
                        importance[feature] += len(rows) * reduction
                current += scale * model.predict(data[:, cols])
            importance = np.maximum(importance, 0)
        else:
            raise ValueError("type must be split or gain")
        total = importance.sum()
        return importance / total if total else importance

    @property
    def feature_importances_(self):
        return self.get_feature_importance()


# Имя из исходного шаблона также поддерживается.
BoostingClassifier = Boosting

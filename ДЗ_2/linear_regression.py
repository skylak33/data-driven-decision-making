import numpy as np
from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface
from descents import AnalyticSolutionOptimizer
from abc import abstractmethod, ABC
from typing import Callable
from scipy import sparse
from scipy.sparse.linalg import svds

def _average_gradient(X, residual_derivative):
    if X.ndim == 2:
        return np.asarray(X.T @ residual_derivative).reshape(-1) / X.shape[0]
    return np.einsum("...nd,...n->...d", X, residual_derivative) / X.shape[-2]


class MSELoss(LossFunction, LossFunctionClosedFormMixin):
    def __init__(self, analytic_solution_func: Callable | None = None):
        self._prepared_X = self._prepared_y = None
        self.analytic_solution_func = (
            self._plain_analytic_solution
            if analytic_solution_func is None else analytic_solution_func
        )

    def prepare(self, X, y):
        self._prepared_X = self._prepared_y = None
        if len(y) > 1000:
            gram = X.T @ X / len(y)
            self._gram = gram.toarray() if sparse.issparse(gram) else gram
            self._cross = np.asarray(X.T @ y).ravel() / len(y)
            self._target_square = float(y @ y / len(y))
            self._prepared_X, self._prepared_y = X, y

    def loss(self, X, y, w) -> float:
        if X is self._prepared_X and y is self._prepared_y:
            return float(np.maximum(0.0, w @ self._gram @ w - 2 * w @ self._cross + self._target_square))
        return float(np.mean((X @ w - y) ** 2))

    def gradient(self, X, y, w) -> np.ndarray:
        if X is self._prepared_X and y is self._prepared_y:
            return 2 * (self._gram @ w - self._cross)
        return _average_gradient(X, 2 * (X @ w - y))

    def analytic_solution(self, X, y) -> np.ndarray:
        return self.analytic_solution_func(X, y)

    @classmethod
    def _plain_analytic_solution(cls, X, y) -> np.ndarray:
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        if np.linalg.matrix_rank(X) < X.shape[1]:
            raise np.linalg.LinAlgError("X is rank deficient; use the SVD solution")
        return np.linalg.solve(X.T @ X, X.T @ y)

    @classmethod
    def _svd_analytic_solution(cls, X, y) -> np.ndarray:
        X = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
        if not np.any(X):
            return np.zeros(X.shape[1])
        r = min(X.shape)
        if r == 1:
            norm_squared = np.sum(X**2)
            return X.T @ y / norm_squared if norm_squared else np.zeros(X.shape[1])
        U, s, Vt = svds(X, k=r - 1, solver="arpack", tol=0, random_state=42)
        cutoff = np.finfo(float).eps * max(X.shape) * s.max(initial=0)
        inv_s = np.divide(1.0, s, out=np.zeros_like(s), where=s > cutoff)
        w = Vt.T @ (inv_s * (U.T @ y))
        residual = X - (U * s) @ Vt
        column = residual[:, np.argmax(np.sum(residual**2, axis=0))]
        norm = np.linalg.norm(column)
        if norm > cutoff:
            u = column / norm
            row = residual.T @ u
            sigma = np.linalg.norm(row)
            if sigma > cutoff:
                w += row * (u @ y) / sigma**2
        return w


class L2Regularization(LossFunction):
    def __init__(self, core_loss: LossFunction, mu_rate=1.0, analytic_solution_func=None):
        if mu_rate < 0:
            raise ValueError("mu_rate must be nonnegative")
        self.core_loss = core_loss
        self.mu_rate = mu_rate
        self.analytic_solution_func = analytic_solution_func

    def prepare(self, X, y):
        self.core_loss.prepare(X, y)

    def loss(self, X, y, w) -> float:
        return self.core_loss.loss(X, y, w) + self.mu_rate / 2 * float(w @ w)

    def gradient(self, X, y, w) -> np.ndarray:
        return self.core_loss.gradient(X, y, w) + self.mu_rate * w


class LogCoshLoss(LossFunction):
    def loss(self, X, y, w) -> float:
        residual = X @ w - y
        return float(np.mean(np.logaddexp(residual, -residual) - np.log(2)))

    def gradient(self, X, y, w) -> np.ndarray:
        return _average_gradient(X, np.tanh(X @ w - y))


class HuberLoss(LossFunction):
    def __init__(self, delta=1.0):
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = delta

    def loss(self, X, y, w) -> float:
        absolute = np.abs(X @ w - y)
        quadratic = np.minimum(absolute, self.delta)
        return float(np.mean(0.5 * quadratic**2 + self.delta * (absolute - quadratic)))

    def gradient(self, X, y, w) -> np.ndarray:
        return _average_gradient(X, np.clip(X @ w - y, -self.delta, self.delta))


class CustomLinearRegression(LinearRegressionInterface):
    def __init__(self, optimizer, loss_function=None):
        self.optimizer = optimizer
        self.optimizer.set_model(self)
        self.loss_function = MSELoss() if loss_function is None else loss_function
        self.loss_history = []
        self.w = self.X_train = self.y_train = None

    def predict(self, X) -> np.ndarray:
        if self.w is None:
            raise ValueError("Fit the model before prediction")
        return X @ self.w

    def _get_data(self, X_batch, y_batch):
        if (X_batch is None) != (y_batch is None):
            raise ValueError("Pass X_batch and y_batch together")
        if X_batch is None:
            return self.X_train, self.y_train
        return X_batch, y_batch

    def compute_gradients(self, X_batch=None, y_batch=None) -> np.ndarray:
        X, y = self._get_data(X_batch, y_batch)
        return self.loss_function.gradient(X, y, self.w)

    def compute_loss(self, X_batch=None, y_batch=None) -> float:
        X, y = self._get_data(X_batch, y_batch)
        return self.loss_function.loss(X, y, self.w)

    def fit(self, X, y) -> None:
        X = X.tocsr().astype(float) if sparse.issparse(X) else np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != len(y) or len(y) == 0:
            raise ValueError("Expected nonempty X of shape (n, d) and y of shape (n,)")
        self.X_train, self.y_train = X, y
        self.w = np.zeros(X.shape[1])
        self.loss_history = []
        self.optimizer.set_model(self)
        self.loss_function.prepare(X, y)
        self.optimizer.optimize()

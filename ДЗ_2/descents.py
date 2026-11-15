import numpy as np
from abc import ABC, abstractmethod
from interfaces import AbstractOptimizer, LearningRateSchedule
from scipy import sparse



class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


class BaseDescent(AbstractOptimizer, ABC):
    def __init__(self, lr_schedule=None, tolerance=1e-6, max_iter=1000):
        self.lr_schedule = TimeDecayLR() if lr_schedule is None else lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.iteration = 0
        self.model = None
        self.stop_reason = None

    def _reset_state(self):
        """"""

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """"""

    def _step(self) -> np.ndarray:
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        if self.model is None:
            raise ValueError("Attach a model before optimization")
        self.iteration = 0
        self._reset_state()
        self.stop_reason = "max_iter"
        self.model.loss_history = [self.model.compute_loss()]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for _ in range(self.max_iter):
                delta = self._step()
                self.model.loss_history.append(self.model.compute_loss())
                if not np.all(np.isfinite(delta)):
                    self.stop_reason = "nonfinite"
                    break
                if delta @ delta < self.tolerance:
                    self.stop_reason = "tolerance"
                    break


class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        delta = -self.lr_schedule.get_lr(self.iteration) * self.model.compute_gradients()
        self.model.w += delta
        return delta


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        indices = np.random.randint(self.model.X_train.shape[0], size=self.batch_size)
        gradient = self.model.compute_gradients(
            self.model.X_train[indices], self.model.y_train[indices]
        )
        delta = -self.lr_schedule.get_lr(self.iteration) * gradient
        self.model.w += delta
        return delta


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self._reset_state()

    def _reset_state(self):
        self.grad_memory = None
        self.grad_sum = None
        self.avg_grad = None

    def _update_weights(self) -> np.ndarray:
        X, y = self.model.X_train, self.model.y_train
        n, d = X.shape
        if self.grad_memory is None:
            self.grad_memory = np.zeros((n, d))
            self.grad_sum = np.zeros(d)
            self.avg_grad = np.zeros(d)
        indices = np.unique(np.random.randint(n, size=self.batch_size))
        batch = X[indices]
        if sparse.issparse(batch):
            batch = batch.toarray()
        # Independent singleton batches: (b, 1, d) -> (b, d), without Python loops.
        gradients = self.model.compute_gradients(batch[:, None, :], y[indices, None])
        self.grad_sum += (gradients - self.grad_memory[indices]).sum(axis=0)
        self.grad_memory[indices] = gradients
        self.avg_grad = self.grad_sum / n
        delta = -self.lr_schedule.get_lr(self.iteration) * self.avg_grad
        self.model.w += delta
        return delta


class MomentumDescent(BaseDescent):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _reset_state(self):
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)
        self.velocity = (
            self.beta * self.velocity
            + self.lr_schedule.get_lr(self.iteration) * self.model.compute_gradients()
        )
        delta = -self.velocity
        self.model.w += delta
        return delta


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = self.v = None

    def _reset_state(self):
        self.m = self.v = None

    def _update_weights(self) -> np.ndarray:
        gradient = self.model.compute_gradients()
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        t = self.iteration + 1
        m_hat = self.m / (1 - self.beta1**t)
        v_hat = self.v / (1 - self.beta2**t)
        delta = -self.lr_schedule.get_lr(self.iteration) * m_hat / (np.sqrt(v_hat) + self.eps)
        self.model.w += delta
        return delta


class AnalyticSolutionOptimizer(AbstractOptimizer):
    def __init__(self):
        self.model = None

    def optimize(self) -> None:
        solve = getattr(self.model.loss_function, "analytic_solution", None)
        if solve is None:
            raise TypeError("This loss has no analytic solution")
        self.model.loss_history = [self.model.compute_loss()]
        self.model.w = solve(self.model.X_train, self.model.y_train)
        self.model.loss_history.append(self.model.compute_loss())

# Parallel-Bayesian-Optimization-Thompson-Sampling is free software:
# you can redistribute it and/or modifyit under the terms of the MIT
# License. You should have received a copy of he MIT License along with
# Parallel-Bayesian-Optimization-Thompson-Sampling.
# If not, see <https://opensource.org/licenses/MIT>.
#

from abc import ABC, abstractmethod
from typing import Union
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import numpy as np


class ThompsonStrategy(ABC, object):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def update(self, reward, action):
        pass

    @property
    def parameters(self):
        return None

    @property
    def mean(self):
        return 0.0

    @property
    def selections(self):
        return None


class BernoulliThompsonStrategy(ThompsonStrategy):
    def __init__(self, num_parameters: int, alpha: list, beta: list):
        super().__init__()
        self.num_parameters = num_parameters

        if (
            not isinstance(alpha, float)
            and not isinstance(alpha, int)
            and len(alpha) != num_parameters
            and len(alpha) != 1
        ):
            raise ValueError("The size of alpha should be {}".format(num_parameters))
        if (
            not isinstance(beta, float)
            and not isinstance(beta, int)
            and len(beta) != num_parameters
            and len(beta) != 1
        ):
            raise ValueError("The size of beta should be {}".format(num_parameters))

        if isinstance(alpha, float) or isinstance(alpha, int):
            self.alpha = np.array([alpha] * num_parameters)
        else:
            self.alpha = np.array(alpha)

        if isinstance(beta, float) or isinstance(beta, int):
            self.beta = np.array([beta] * num_parameters)
        else:
            self.beta = np.array(beta)

        self.num_selections = np.array([0] * num_parameters)

    def sample(self) -> int:
        arm = np.random.beta(self.alpha, self.beta, size=self.num_parameters).argmax()
        self.num_selections[arm] += 1
        return arm

    def update(self, reward: float, action: int):
        self.alpha[action] = max(self.alpha[action] + reward, 1)
        self.beta[action] = max(self.beta[action] + 1 - reward, 1)

    @property
    def parameters(self):
        return (self.alpha, self.beta)

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def selections(self):
        return self.num_selections


class GaussianProcessesStrategy(ThompsonStrategy):
    def __init__(
        self,
        num_parameters: int,
        interval: Union[np.ndarray, "list[np.ndarray]"],
        beta: float = 1.0,
        mu_init: float = 0.0,
        sigma_init: float = 0.5,
        gp_regressor=None,
        fit_threshold: int = 25,
        update_freq: int = 5,
    ):
        super().__init__()
        self.num_parameters = num_parameters
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        self.fit_threshold = fit_threshold
        self.update_freq = update_freq
        if isinstance(interval, list):
            if len(interval) != self.num_parameters:
                raise ValueError(
                    "When passing a list the length should be the same as num_parameters"
                )
            else:
                self.mesh = np.meshgrid(*interval)

        elif isinstance(interval, np.ndarray):
            self.mesh = np.meshgrid(*[interval] * self.num_parameters)

        self.X_grid = np.column_stack(list(map(np.ravel, self.mesh)))
        if gp_regressor is None:
            kernel = C(1, (1e-3, 1e1)) * RBF(1, (1e-2, 1e1)) + WhiteKernel()
            self.gp = GaussianProcessRegressor(kernel=kernel)
        else:
            self.gp = gp_regressor

        self.mu: np.ndarray = np.array([self.mu_init] * self.X_grid.shape[0])
        self.sigma: np.ndarray = np.array([self.sigma_init] * self.X_grid.shape[0])
        self.beta = beta
        self.t: int = 0  # Number of times we have sampled with gp-ucb
        self.N: np.ndarray = np.zeros_like(
            self.mu
        )  # number of times we sampled a given X for time t
        self.X: list = []
        self.y: list = []

    def sample(self):
        if len(self.X) < self.fit_threshold:
            grid_idx = np.random.randint(self.X_grid.shape[0])
        else:
            self.t += 1
            beta = self.beta * np.sqrt(1 + self.t * np.log(self.t) ** 2 / np.maximum(self.N, 1))
            grid_idx = (self.mu.flatten() + self.sigma.flatten() * np.sqrt(beta)).argmax()

        self.N[grid_idx] += 1
        params = self.X_grid[grid_idx]
        return params

    def update(self, reward, params):
        self.X.append(params)
        self.y.append(reward)
        if len(self.X) < self.fit_threshold or len(self.X) % self.update_freq != 0:
            return
        self.gp = self.gp.fit(self.X, self.y)
        self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)

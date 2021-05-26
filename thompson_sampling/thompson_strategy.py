# Parallel-Bayesian-Optimization-Thompson-Sampling is free software:
# you can redistribute it and/or modifyit under the terms of the MIT
# License. You should have received a copy of he MIT License along with
# Parallel-Bayesian-Optimization-Thompson-Sampling.
# If not, see <https://opensource.org/licenses/MIT>.
#

from abc import ABC, abstractmethod
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
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
        interval: "np.ndarray | list[np.ndarray]",
        beta: float,
    ):
        super().__init__()
        self.num_parameters = num_parameters
        if isinstance(interval, list):
            if len(interval) != self.num_parameters:
                raise ValueError(
                    "When passing a list the length should be the same as num_parameters"
                )
            else:
                mesh = np.meshgrid(*interval)

        elif isinstance(interval, np.ndarray):
            mesh = np.meshgrid(*[interval for _ in range(self.num_parameters)])

        mapfunc = partial(np.ravel, order="A")
        self.X_grid = np.column_stack(list(map(mapfunc, mesh)))

        self.mu = np.array([0.0 for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([np.random.randn() for _ in range(self.X_grid.shape[0])])
        self.beta = beta
        self.gp = GaussianProcessRegressor()
        self.X = np.array([])
        self.y = np.array([])

    def sample(self):
        # Fit the GP to the observations we have
        grid_idx = (self.mu + self.sigma * np.sqrt(self.beta)).argmax()
        params = self.X_grid[grid_idx]
        return params

    def update(self, reward, action):
        self.X = np.append(self.X, action)
        self.y = np.append(self.y, reward)
        if self.X.shape[0] < 25:
            return
        self.gp = GaussianProcessRegressor()
        self.gp = self.gp.fit(self.X.reshape(-1, 1), self.y)
        self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)

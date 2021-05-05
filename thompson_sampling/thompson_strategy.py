# Parallel-Bayesian-Optimization-Thompson-Sampling is free software:
# you can redistribute it and/or modifyit under the terms of the MIT
# License. You should have received a copy of he MIT License along with
# Parallel-Bayesian-Optimization-Thompson-Sampling.
# If not, see <https://opensource.org/licenses/MIT>.
#

import numpy as np


class ThompsonStrategy(object):
    def __init__(self):
        pass

    def sample(self):
        pass

    def update(self, reward, action):
        pass

    @property
    def parameters(self):
        return None

    @property
    def mean(self):
        return 0.

    @property
    def selections(self):
        return None


class BernoulliThompsonStrategy(ThompsonStrategy):
    def __init__(self, num_parameters: int, alpha: list, beta: list):
        super().__init__()
        self.num_parameters = num_parameters

        if not isinstance(alpha, float) and not isinstance(alpha, int) \
                and len(alpha) != num_parameters and len(alpha) != 1:
            raise ValueError(
                'The size of alpha should be {}'.format(num_parameters))
        if not isinstance(beta, float) and not isinstance(beta, int)  \
                and len(beta) != num_parameters and len(beta) != 1:
            raise ValueError(
                'The size of beta should be {}'.format(num_parameters))

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
        arm = np.random.beta(
            self.alpha, self.beta, size=self.num_parameters).argmax()
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

# Parallel-Bayesian-Optimization-Thompson-Sampling is free software:
# you can redistribute it and/or modifyit under the terms of the MIT
# License. You should have received a copy of he MIT License along with
# Parallel-Bayesian-Optimization-Thompson-Sampling.
# If not, see <https://opensource.org/licenses/MIT>.
#

import multiprocessing as mp
import numpy as np
from functools import reduce
from copy import deepcopy
from .thompson_strategy import ThompsonStrategy


class ThompsonSampling(object):
    def __init__(
        self,
        thompson_strategy: ThompsonStrategy,
        epochs: int,
        fitness_function: callable,
        callbacks: dict = {},
        num_processors: int = 1,
    ):

        self.__callbacks = ["on_update"]

        if np.any([x not in self.__callbacks for x in callbacks.keys()]):
            raise ValueError("One of the callbacks is not available")

        if not isinstance(thompson_strategy, ThompsonStrategy):
            raise ValueError("Invalid thompson strategy.")

        self.epochs = int(epochs)
        self.strategy = thompson_strategy

        self.fitness_function = fitness_function

        self.callbacks = callbacks
        self.num_processors = num_processors

    def run(self):
        results = []
        epoch = 0.0

        with mp.Pool(processes=self.num_processors) as pool:
            for p in range(self.num_processors):
                r = pool.apply_async(self.fitness_function, (self.strategy.sample(),))
                results.append(r)

            while epoch < self.epochs:
                r = results.pop(0)
                if r.ready():
                    x = r.get()
                    self.strategy.update(*x)
                    if "on_update" in self.callbacks:
                        if self.callbacks["on_update"](
                            epoch, self.strategy, epoch == self.epochs - 1
                        ):
                            break

                    results.append(
                        pool.apply_async(self.fitness_function, (self.strategy.sample(),))
                    )
                    epoch += 1 / self.num_processors
                else:
                    results.append(r)

        return self.strategy

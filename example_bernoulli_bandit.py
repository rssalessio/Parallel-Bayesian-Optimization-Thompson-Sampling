# Parallel-Bayesian-Optimization-Thompson-Sampling is free software:
# you can redistribute it and/or modifyit under the terms of the MIT
# License. You should have received a copy of he MIT License along with
# Parallel-Bayesian-Optimization-Thompson-Sampling.
# If not, see <https://opensource.org/licenses/MIT>.
#

import sys
import numpy as np
from thompson_sampling import ThompsonSampling, BernoulliThompsonStrategy

sys.path.append("..")

# Example Bernolli Bandit
# ------------
# In this example we see how to use Thompson Sampling
# to solve the Bernoulli Bandit problem


def fitness(arm: int):
    bernoulli_parameters = [0.1, 0.3, 0.9, 0.4]

    if np.random.rand() < bernoulli_parameters[arm]:
        return 1, arm
    else:
        return 0, arm


def on_update(epoch, strategy, last_epoch):
    if epoch % 10 == 0:
        print("Epoch {} - Parameters mean {}".format(epoch, strategy.mean))
    if np.any(strategy.mean > 0.9) or last_epoch:
        print("Search is over!")
        print("Epoch {} - Parameters mean {}".format(epoch, strategy.mean))
        print("Epoch {} - Number of times an arm as selected {}".format(epoch, strategy.selections))
        return True
    return False


if __name__ == "__main__":
    strategy = BernoulliThompsonStrategy(4, 1, 1)
    ts = ThompsonSampling(
        thompson_strategy=strategy,
        epochs=100,
        fitness_function=fitness,
        callbacks={"on_update": on_update},
        num_processors=1,
    )
    strategy = ts.run()

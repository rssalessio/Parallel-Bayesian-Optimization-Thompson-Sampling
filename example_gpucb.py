import numpy as np

from thompson_sampling import GaussianProcessesStrategy, ThompsonSampling


def fitness(params: np.ndarray):
    return np.exp(-(params ** 2)), params


def on_update(epoch, strategy, last_epoch):
    if epoch % 10 == 0:
        print("Epoch {} - Current sample {}".format(epoch, strategy.sample()))
    return False


if __name__ == "__main__":
    strategy = GaussianProcessesStrategy(1, np.arange(-10, 10, 0.1), 1.0)
    ts = ThompsonSampling(
        thompson_strategy=strategy,
        epochs=500,
        fitness_function=fitness,
        callbacks={"on_update": on_update},
    )
    strategy = ts.run()
    print(strategy.sample())

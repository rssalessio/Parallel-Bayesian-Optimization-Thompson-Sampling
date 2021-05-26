import numpy as np

from thompson_sampling import GaussianProcessesStrategy, ThompsonSampling


def fitness(params: np.ndarray):
    reward = np.exp(-(params ** 2)) + np.cos(params) + np.random.normal()
    return reward, params


def on_update(epoch, strategy, last_epoch):
    if epoch % 10 == 0:
        print("Epoch {} - Current sample {}".format(epoch, strategy.sample()))
    if last_epoch:
        print("SEARCH DONE")
        print(strategy.sample())
    return False


if __name__ == "__main__":
    strategy = GaussianProcessesStrategy(1, np.arange(-10, 10, 0.1), 1.0)
    ts = ThompsonSampling(
        thompson_strategy=strategy,
        epochs=200,
        fitness_function=fitness,
        callbacks={"on_update": on_update},
        num_processors=1,
    )
    strategy = ts.run()

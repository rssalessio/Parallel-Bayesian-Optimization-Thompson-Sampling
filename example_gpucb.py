import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from thompson_sampling import GaussianProcessesStrategy, ThompsonSampling


def fitness(params: np.ndarray):
    reward = (
        np.exp(-(params ** 2)) + np.cos(params) + np.random.normal(scale=0.1, size=params.shape)
    )
    return reward, params


def on_update(epoch, strategy, last_epoch):
    if epoch % 10 == 0:
        print("Epoch {} - Current sample {}".format(epoch, strategy.sample()))
    if last_epoch:
        print("SEARCH DONE")
        print(strategy.sample())
    return False


def plot1d(strategy):
    data = np.exp(-(strategy.X_grid ** 2)) + np.cos(strategy.X_grid)
    plt.plot(strategy.X_grid, data, label="Ground truth")
    plt.plot(strategy.X_grid, strategy.mu, "r-*", label="mu")
    plt.plot(strategy.X_grid, strategy.sigma, label="sigma")
    plt.legend()
    plt.show()


def plot(strategy):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(
        strategy.mesh[0],
        strategy.mesh[1],
        strategy.mu.reshape(strategy.mesh[0].shape),
        alpha=0.5,
        color="g",
    )

    ax.plot_wireframe(
        strategy.mesh[0],
        strategy.mesh[1],
        fitness(strategy.mesh)[0],
        alpha=0.5,
        color="b",
    )
    ax.scatter(
        [x[0] for x in strategy.X],
        [x[1] for x in strategy.X],
        strategy.y,
        c="r",
        marker="o",
        alpha=1.0,
    )
    plt.show()
    plt.savefig("fig_%02d.png" % len(strategy.X))


if __name__ == "__main__":
    strategy = GaussianProcessesStrategy(1, np.arange(-10, 10, 0.1), 1.0)
    ts = ThompsonSampling(
        thompson_strategy=strategy,
        epochs=50,
        fitness_function=fitness,
        callbacks={"on_update": on_update},
        num_processors=10,
    )
    strategy = ts.run()
    plot1d(strategy)

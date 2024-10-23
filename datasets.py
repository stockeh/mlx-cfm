import numpy as np
import mlx.core as mx


def multinomial(input, num_samples, replacement=False):
    input = np.array(input, dtype=np.float64)
    if np.any(input < 0):
        raise ValueError("Input contains negative values.")
    input_sum = np.sum(input, axis=-1, keepdims=True)
    if np.any(input_sum == 0):
        raise ValueError(
            "Input contains rows with zero sum, which cannot be used for sampling.")

    probabilities = input / input_sum

    return np.random.choice(len(input), size=num_samples, p=probabilities, replace=replacement)


def eight_normal_sample(n, dim, scale=1, var=1):
    centers = mx.array([
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / mx.sqrt(2), 1.0 / mx.sqrt(2)),
        (1.0 / mx.sqrt(2), -1.0 / mx.sqrt(2)),
        (-1.0 / mx.sqrt(2), 1.0 / mx.sqrt(2)),
        (-1.0 / mx.sqrt(2), -1.0 / mx.sqrt(2)),
    ]) * scale

    noise = mx.random.multivariate_normal(
        mean=mx.zeros(dim), cov=mx.sqrt(var)*mx.eye(dim), shape=(n,), stream=mx.cpu
    )

    multi = mx.array(multinomial(mx.ones(8), n, replacement=True))
    return centers[multi] + noise


def generate_moons(n_samples: int = 100, noise: float = 1e-4):
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_linspace = mx.linspace(0, mx.pi, n_samples_out)
    inner_linspace = mx.linspace(0, mx.pi, n_samples_in)

    outer_circ_x = mx.cos(outer_linspace)
    outer_circ_y = mx.sin(outer_linspace)

    inner_circ_x = 1 - mx.cos(inner_linspace)
    inner_circ_y = 1 - mx.sin(inner_linspace) - 0.5

    X = mx.zeros((n_samples, 2))
    X[:n_samples_out, 0] = outer_circ_x
    X[:n_samples_out, 1] = outer_circ_y
    X[n_samples_out:, 0] = inner_circ_x
    X[n_samples_out:, 1] = inner_circ_y

    T = mx.concatenate([mx.zeros(n_samples_out, dtype=mx.int16),
                       mx.ones(n_samples_in, dtype=mx.int16)])

    if noise is not None:
        X += mx.random.uniform(shape=(n_samples, 2)) * noise

    return X, T


def sample_moons(n):
    X, T = generate_moons(n, noise=0.2)
    X = X * 2.6 - 1
    X[:, 0] -= 0.45
    return X, T


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 2048
    colors = ['#1b9e77', '#d95f02']
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    X = sample_8gaussians(n)
    ax.scatter(X[:, 0], X[:, 1], s=4, color='k', alpha=0.5)

    X, T = sample_moons(n)
    ax.scatter(X[:, 0], X[:, 1], s=2, color=[colors[t.item()]
               for t in T], alpha=0.8)

    fig.tight_layout()
    fig.savefig('media/moons.png', dpi=300, bbox_inches='tight')
    plt.show()

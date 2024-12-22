import time
import argparse
import numpy as np
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim

from tqdm import tqdm
from functools import partial
from typing import Callable

from models import MLP
from cfm import str_to_cfm
from odeint import NeuralODE
from data import sample_moons, sample_8gaussians

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "-m", "--method", type=str, default="vp", choices=["cfm", "vp"], help="cfm method"
)
parser.add_argument("--sigma", type=float, default=0.1, help="sigma for cfm")
parser.add_argument(
    "--solver",
    type=str,
    default="euler",
    choices=["dopri5", "euler"],
    help="ODE solver",
)
parser.add_argument("-b", "--batch_size", type=int, default=512, help="batch size")
parser.add_argument("-e", "--steps", type=int, default=20000, help="number of steps")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


class NeuralNetwork:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        flow_matcher,
        integrator,
        prior_data: Callable,
        target_data: Callable,
    ):
        self.model = model
        self.optimizer = optimizer
        self.flow_matcher = flow_matcher
        self.integrator = integrator

        # data functions
        self.prior_data = prior_data
        self.target_data = target_data

        # bookkeeping
        self.train_error_trace = []

    def eval_fn(self, X, T):
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(X, T)
        vt = self.model(t, xt, repeat=False)
        loss = nn.losses.mse_loss(vt, ut, reduction="mean")
        return loss

    def train(self, steps: int, batch_size: int):
        state = [self.model.state, self.optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(X, T):
            train_step_fn = nn.value_and_grad(self.model, self.eval_fn)
            loss, grads = train_step_fn(X, T)
            self.optimizer.update(self.model, grads)
            return loss

        step_bar = tqdm(range(steps), desc="Training", unit="step")
        self.model.train()
        for _ in step_bar:
            X = self.prior_data(batch_size)
            T, _ = self.target_data(batch_size)

            loss = step(X, T)
            mx.eval(state)

            self.train_error_trace.append(loss.item())
            postfix = {"loss": f"{loss.item():.3f}"}
            step_bar.set_postfix(postfix)

    def use(self, xs=1024, ts=128):
        start_t = time.time()
        self.model.eval()
        traj = self.integrator.trajectory(
            self.prior_data(xs), t_span=mx.linspace(0, 1, ts)
        )
        print(f"Sampled: {time.time() - start_t:.2f} sec")
        return traj


def main(args):
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    kwargs = {
        "n_inputs": 2,
        "n_hiddens_list": [64] * 2,
        "n_outputs": 2,
        "activation_f": "selu",
        "time_varying": True,
    }
    model = MLP(**kwargs)
    model.summary()

    flow_matcher = str_to_cfm(args.method, sigma=args.sigma)
    print(flow_matcher)
    integrator = NeuralODE(model, solver=args.solver, atol=1e-4, rtol=1e-4)
    print(integrator)

    optimizer = optim.AdamW(learning_rate=args.lr)
    net = NeuralNetwork(
        model, optimizer, flow_matcher, integrator, sample_8gaussians, sample_moons
    )

    net.train(args.steps, args.batch_size)
    traj = net.use(xs=1024, ts=128)

    #! plotting
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KernelDensity

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.scatter(
        traj[0, :, 0],
        traj[0, :, 1],
        s=10,
        alpha=0.8,
        c="k",
        label=r"$\mathbf{x}_0 \sim \pi(\mathbf{x})$",
    )
    ax.scatter(traj[:, :, 0], traj[:, :, 1], s=0.2, alpha=0.15, c="k")
    ax.scatter(
        traj[-1, :, 0],
        traj[-1, :, 1],
        s=4,
        alpha=1,
        c="blue",
        label=r"$\mathbf{x}_1 \approx \mathbf{x}^\prime\;[\sim p(\mathbf{x})]$",
    )

    def plot_kde(data, x_grid, y_grid, ax, bandwidth=0.3, cmap="Blues", alpha=0.5):
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(data)

        grid_samples = mx.stack([mx.flatten(x_grid), mx.flatten(y_grid)]).T

        log_density = kde.score_samples(grid_samples)
        density = mx.exp(log_density).reshape(x_grid.shape)
        ax.contourf(x_grid, y_grid, density, levels=100, cmap=cmap, alpha=alpha)

    prior_data = traj[0, :, :2]  # Prior sample z(S)
    last_data = traj[-1, :, :2]  # Last sample z(0)

    x_min = traj[:, :, 0].min() - 1
    x_max = traj[:, :, 0].max() + 1
    y_min = traj[:, :, 1].min() - 1
    y_max = traj[:, :, 1].max() + 1
    x_grid, y_grid = mx.meshgrid(
        mx.linspace(x_min, x_max, 100), mx.linspace(y_min, y_max, 100)
    )

    plot_kde(prior_data, x_grid, y_grid, ax, bandwidth=0.5, cmap="Reds", alpha=0.2)
    plot_kde(last_data, x_grid, y_grid, ax, bandwidth=0.5, cmap="Purples", alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{flow_matcher.name} ({args.solver})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        f"media/trajectories_{args.method}_{args.solver}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    main(args)

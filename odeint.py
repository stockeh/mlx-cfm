import mlx.core as mx
from solver import str_to_solver


def hairer_norm(arr):
    """L2-type norm used for stepsize selection."""
    return mx.sqrt(mx.mean(mx.power(mx.abs(arr), 2)))


def init_step(f, f0, x0, t0, order, atol, rtol):
    """
    Estimate a good initial dt by comparing the scale of x0 and f0.
    """
    scale = atol + mx.abs(x0) * rtol
    d0 = hairer_norm(x0 / scale)
    d1 = hairer_norm(f0 / scale)

    if (d0 < 1e-5) or (d1 < 1e-5):
        h0 = mx.array(1e-6, dtype=t0.dtype)
    else:
        h0 = 0.01 * d0 / d1

    x_new = x0 + h0 * f0
    f_new = f(t0 + h0, x_new)

    d2 = hairer_norm((f_new - f0) / scale) / h0
    if (d1 <= 1e-15) and (d2 <= 1e-15):
        h1 = mx.maximum(mx.array(1e-6, dtype=t0.dtype), h0 * 1e-3)
    else:
        h1 = (0.01 / mx.maximum(d1, d2)) ** (1.0 / float(order + 1))

    dt = mx.minimum(100 * h0, h1)
    return dt


def adapt_step(dt, error_ratio, safety, min_factor, max_factor, order):
    """
    Adaptive stepsize update. If error_ratio < 1, we accept and possibly increase dt.
    If error_ratio > 1, we reject and decrease dt.
    """
    if error_ratio == 0:
        return dt * max_factor
    if error_ratio < 1:
        min_factor = mx.ones_like(dt)
    exponent = mx.array(order, dtype=dt.dtype).reciprocal()
    factor = mx.minimum(
        max_factor, mx.maximum(safety / error_ratio**exponent, min_factor)
    )
    return dt * factor


def adaptive_odeint(
    f,
    k1,
    x,
    dt,
    t_span,
    solver,
    atol=1e-4,
    rtol=1e-4,
    args=None,
    interpolator=None,
    return_all_eval=False,
    seminorm=(False, None),
):
    """
    Adaptive integrator for solvers that provide an error estimate (DormandPrince45, etc).
    """
    # t, T as scalars
    t = t_span[0]
    T = t_span[-1]
    # t_eval for checkpoint times
    t_eval = t_span[1:] if len(t_span) > 1 else mx.array([], dtype=t_span.dtype)

    ckpt_counter, ckpt_flag = 0, False
    eval_times, sol = [t], [x]

    while t < T:
        # Adjust dt if we're about to overshoot T
        if t + dt > T:
            dt = T - t

        # Possibly shorten dt to hit a checkpoint exactly
        if len(t_eval) > 0 and (ckpt_counter < len(t_eval)):
            if interpolator is None:
                next_ckpt = t_eval[ckpt_counter]
                if t + dt > next_ckpt:
                    dt_old = dt
                    dt = next_ckpt - t
                    ckpt_flag = True

        # Step
        f_new, x_new, x_err, stages = solver.step(f, x, t, dt, k1, args=args)

        # Error ratio
        if x_err is not None:
            if seminorm[0]:
                sd = seminorm[1]
                error_scaled = x_err[:sd] / (
                    atol + rtol * mx.maximum(mx.abs(x[:sd]), mx.abs(x_new[:sd]))
                )
            else:
                error_scaled = x_err / (
                    atol + rtol * mx.maximum(mx.abs(x), mx.abs(x_new))
                )
            error_ratio = hairer_norm(error_scaled)
        else:
            # No error => accept
            error_ratio = 0

        accept_step = error_ratio <= 1

        if accept_step:
            # Interpolation
            if interpolator is not None and (ckpt_counter < len(t_eval)):
                coefs = None
                while (ckpt_counter < len(t_eval)) and (
                    (t + dt) > t_eval[ckpt_counter]
                ):
                    t0, t1 = t, t + dt
                    if coefs is None:
                        x_mid = x + dt * sum(
                            interpolator.bmid[i] * stages[i] for i in range(len(stages))
                        )
                        f0, f1, x0, x1 = k1, f_new, x, x_new
                        coefs = interpolator.fit(dt, f0, f1, x0, x1, x_mid)
                    x_ckpt = interpolator.evaluate(coefs, t0, t1, t_eval[ckpt_counter])
                    sol.append(x_ckpt)
                    eval_times.append(t_eval[ckpt_counter])
                    ckpt_counter += 1

            # If new time matches a checkpoint or we want all evaluations
            if (
                (ckpt_counter < len(t_eval))
                and (mx.isclose(t + dt, t_eval[ckpt_counter]).sum() == 1)
            ) or return_all_eval:
                sol.append(x_new)
                eval_times.append(t + dt)
                if (ckpt_counter < len(t_eval)) and mx.isclose(
                    t + dt, t_eval[ckpt_counter]
                ).sum():
                    ckpt_counter += 1

            # Accept
            t, x = t + dt, x_new
            k1 = f_new
        else:
            # Revert dt if we shortened for a checkpoint
            if ckpt_flag:
                dt = dt_old
        ckpt_flag = False

        # Adapt dt
        dt = adapt_step(
            dt,
            error_ratio,
            solver.safety,
            solver.min_factor,
            solver.max_factor,
            solver.order,
        )

    return mx.array(eval_times), mx.stack(sol)


def fixed_odeint(f, x, t_span, solver, save_at=None, args=None):
    """
    Fixed-step integrator (Euler, Midpoint, etc).
    """
    if save_at is None:
        save_at = t_span
    elif not isinstance(save_at, mx.array):
        save_at = mx.array(save_at)

    # Each save_at time must appear exactly once in t_span
    for t_s in save_at:
        c = mx.isclose(t_span, t_s).sum()
        assert c == 1, f"Time {t_s} in save_at not found exactly once in t_span!"

    t = t_span[0]
    T = t_span[-1]
    dt = t_span[1] - t_span[0] if len(t_span) > 1 else mx.array(0.0, dtype=t.dtype)

    sol = []
    if mx.isclose(t, save_at).sum():
        sol.append(x)

    steps = 0
    while steps < len(t_span) - 1:
        # Step
        f_new, x_new, err, stages = solver.step(f, x, t, dt, k1=None, args=args)
        x = x_new
        t = t + dt
        steps += 1

        # If new time is in save_at, store
        if mx.isclose(t, save_at).sum():
            sol.append(x)

        # Prepare next dt
        if steps < len(t_span) - 1:
            dt = t_span[steps + 1] - t

    if isinstance(sol[0], dict):
        # If the state is a dictionary
        final_out = {k: [] for k in sol[0].keys()}
        for s in sol:
            for k in s.keys():
                final_out[k].append(s[k])
        final_out = {k: mx.stack(v) for k, v in final_out.items()}
        return save_at, final_out

    return save_at, mx.stack(sol)


def odeint(f, x, t_span, solver, atol: float = 1e-3, rtol: float = 1e-3):
    """
    High-level ODE solver interface. Uses fixed or adaptive integration.
    """
    if not isinstance(t_span, mx.array):
        t_span = mx.array(t_span)

    if solver.stepping_class == "fixed":
        return fixed_odeint(f, x, t_span, solver)
    else:
        t0 = t_span[0]
        k1 = f(t0, x)
        dt = init_step(f, k1, x, t0, solver.order, atol, rtol)
        return adaptive_odeint(f, k1, x, dt, t_span, solver, atol, rtol)


################################################################################
#                                  NEURAL ODE                                  #
################################################################################


class NeuralODE:
    """
    Example container that uses 'odeint' to integrate a vector field.
    """

    def __init__(self, vector_field, solver="dopri5", atol=1e-4, rtol=1e-4):
        self.vf = vector_field
        self.solver = str_to_solver(solver)
        self.atol = atol
        self.rtol = rtol

    def forward(self, x, t_span):
        """Can define your own forward pass if needed."""
        raise NotImplementedError

    def trajectory(self, x, t_span):
        """
        Integrate the vector_field from x(t0) across t_span.
        Returns the stacked solution states.
        """
        _, sol = odeint(self.vf, x, t_span, self.solver, self.atol, self.rtol)
        return sol

    def __repr__(self):
        return (
            f"Neural ODE:\n\t- solver: {self.solver}"
            f"\n\t- order: {self.solver.order}"
            f"\n\t- tolerances: relative {self.rtol} absolute {self.atol}"
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def vector_field(t, x):  # dx/dt = -x
        return -x

    x0 = mx.array([[1.0]])
    t_span = mx.linspace(0, 5, 50)

    euler_ode = NeuralODE(vector_field, solver="euler")
    midpoint_ode = NeuralODE(vector_field, solver="midpoint")
    dopri_ode = NeuralODE(vector_field, solver="dopri5", atol=1e-6, rtol=1e-6)

    sol_euler = euler_ode.trajectory(x0, t_span)
    sol_midpoint = midpoint_ode.trajectory(x0, t_span)
    sol_dopri = dopri_ode.trajectory(x0, t_span)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(t_span, sol_euler[:, 0, 0], ".-", label="NeuralODE Euler")
    ax.plot(t_span, sol_midpoint[:, 0, 0], "+-", label="NeuralODE Midpoint")
    ax.plot(t_span, sol_dopri[:, 0, 0], "--", label="NeuralODE Dopri5")

    ax.set_xlabel("time")
    ax.set_ylabel("x(t)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

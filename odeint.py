import mlx.core as mx

from solver import str_to_solver


def hairer_norm(arr):
    return mx.sqrt(mx.mean(mx.power(mx.abs(arr), 2)))


def init_step(f, f0, x0, t0, order, atol, rtol):
    scale = atol + mx.abs(x0) * rtol
    d0, d1 = hairer_norm(x0 / scale), hairer_norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = mx.array(1e-6, dtype=t0.dtype)
    else:
        h0 = 0.01 * d0 / d1

    x_new = x0 + h0 * f0
    f_new = f(t0 + h0, x_new)

    d2 = hairer_norm((f_new - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = mx.maximum(mx.array(1e-6, dtype=t0.dtype), h0 * 1e-3)
    else:
        h1 = (0.01 / mx.maximum(d1, d2)) ** (1.0 / float(order + 1))
    dt = mx.minimum(100 * h0, h1)
    return dt


def adapt_step(dt, error_ratio, safety, min_factor, max_factor, order):
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
    t_eval, t, T = t_span[1:], t_span[:1], t_span[-1]
    ckpt_counter, ckpt_flag = 0, False
    eval_times, sol = [t], [x]
    while t < T:
        if t + dt > T:
            dt = T - t
        ############### checkpointing ###############################
        if t_eval is not None:
            # satisfy checkpointing by using interpolation scheme or resetting `dt`
            if (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
                if interpolator == None:
                    # save old dt, raise "checkpoint" flag and repeat step
                    dt_old, ckpt_flag = dt, True
                    dt = t_eval[ckpt_counter] - t

        f_new, x_new, x_err, stages = solver.step(f, x, t, dt, k1, args=args)
        ################# compute error #############################
        if seminorm[0] == True:
            state_dim = seminorm[1]
            error = x_err[:state_dim]
            error_scaled = error / (
                atol
                + rtol * mx.maximum(mx.abs(x[:state_dim]), mx.abs(x_new[:state_dim]))
            )
        else:
            error = x_err
            error_scaled = error / (atol + rtol * mx.maximum(mx.abs(x), mx.abs(x_new)))
        error_ratio = hairer_norm(error_scaled)
        accept_step = error_ratio <= 1

        if accept_step:
            ############### checkpointing via interpolation ###############################
            if t_eval is not None and interpolator is not None:
                coefs = None
                while (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
                    t0, t1 = t, t + dt
                    x_mid = x + dt * sum(
                        [interpolator.bmid[i] * stages[i] for i in range(len(stages))]
                    )
                    f0, f1, x0, x1 = k1, f_new, x, x_new
                    if coefs == None:
                        coefs = interpolator.fit(dt, f0, f1, x0, x1, x_mid)
                    x_in = interpolator.evaluate(coefs, t0, t1, t_eval[ckpt_counter])
                    sol.append(x_in)
                    eval_times.append(t_eval[ckpt_counter][None])
                    ckpt_counter += 1

            if t + dt == t_eval[ckpt_counter] or return_all_eval:  # note (1)
                sol.append(x_new)
                eval_times.append(t + dt)
                # we only increment the ckpt counter if the solution points corresponds to a time point in `t_span`
                if t + dt == t_eval[ckpt_counter]:
                    ckpt_counter += 1
            t, x = t + dt, x_new
            k1 = f_new

        ################ stepsize control ###########################
        # reset "dt" in case of checkpoint without interp
        if ckpt_flag:
            dt = dt_old - dt
            ckpt_flag = False

        dt = adapt_step(
            dt,
            error_ratio,
            solver.safety,
            solver.min_factor,
            solver.max_factor,
            solver.order,
        )

    return mx.concatenate(eval_times), mx.stack(sol)


def fixed_odeint(f, x, t_span, solver, save_at=(), args=None):
    """Solves IVPs with same `t_span`, using fixed-step methods"""
    if len(save_at) == 0:
        save_at = t_span
    if not isinstance(save_at, mx.array):
        save_at = mx.array(save_at)

    assert all(
        mx.isclose(t, save_at).sum() == 1 for t in save_at
    ), "each element of save_at [torch.Tensor] must be contained in t_span [torch.Tensor] once and only once"

    t, T, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

    sol = []
    if mx.isclose(t, save_at).sum():
        sol = [x]

    steps = 1
    while steps <= len(t_span) - 1:
        _, x, _ = solver.step(f, x, t, dt, k1=None, args=args)
        t = t + dt

        if mx.isclose(t, save_at).sum():
            sol.append(x)
        if steps < len(t_span) - 1:
            dt = t_span[steps + 1] - t
        steps += 1

    if isinstance(sol[0], dict):
        final_out = {k: [v] for k, v in sol[0].items()}
        _ = [final_out[k].append(x[k]) for k in x.keys() for x in sol[1:]]
        final_out = {k: mx.stack(v) for k, v in final_out.items()}
    elif isinstance(sol[0], mx.array):
        final_out = mx.stack(sol)
    else:
        raise NotImplementedError(f"{type(x)} is not supported as the state variable")

    return save_at, final_out


def odeint(f, x, t_span, solver, atol: float = 1e-3, rtol: float = 1e-3):
    if solver.stepping_class == "fixed":
        return fixed_odeint(f, x, t_span, solver, args=None)

    t = t_span[0]
    k1 = f(t, x)
    dt = init_step(f, k1, x, t, solver.order, atol, rtol)
    return adaptive_odeint(f, k1, x, dt, t_span, solver, atol, rtol, args=None)


class NeuralODE:
    def __init__(self, vector_field, solver="dopri5", atol=1e-4, rtol=1e-4):
        self.vf = vector_field
        self.solver = str_to_solver(solver)
        self.atol = atol
        self.rtol = rtol

    def forward(self, x, t_span):
        raise NotImplementedError

    def trajectory(self, x, t_span):
        _, sol = odeint(self.vf, x, t_span, self.solver, atol=self.atol, rtol=self.rtol)
        return sol

    def __repr__(self):
        return f"Neural ODE:\n\t- solver: {self.solver}\
        \n\t- order: {self.solver.order}\
        \n\t- tolerances: relative {self.rtol} absolute {self.atol}"


if __name__ == "__main__":

    def vector_field(t, x):
        # (dx/dt = -x)
        return -x

    x0 = mx.array([[1.0]])
    t_span = mx.linspace(0, 5, 100)

    ode_solver = NeuralODE(vector_field)

    solution = ode_solver.trajectory(x0, t_span)

    #! plotting
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(t_span, solution[:, 0, 0], lw=2, color="k")
    ax.set_xlabel("Time")
    ax.set_ylabel("x(t)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()

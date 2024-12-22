from typing import Tuple
import mlx.core as mx


class SolverTemplate:
    def __init__(
        self,
        order,
        min_factor: float = 0.2,
        max_factor: float = 10,
        safety: float = 0.9,
    ):
        """
        Parameters
        ----------
        order : int
            The numerical order of the ODE solver.
        min_factor : float, optional
            Minimum safety factor for adaptive stepsize control.
        max_factor : float, optional
            Maximum safety factor for adaptive stepsize control.
        safety : float, optional
            Typical safety factor, e.g., 0.9.
        """
        super().__init__()
        self.order = order
        self.min_factor = mx.array([min_factor])
        self.max_factor = mx.array([max_factor])
        self.safety = mx.array([safety])

    def step(self, f, x, t, dt, k1=None, args=None):
        """
        Perform one ODE integration step.

        Parameters
        ----------
        f : callable
            The vector field, i.e. f(t, x).
        x : mx.array or similar
            Current state at time t.
        t : mx.array or float
            Current time.
        dt : mx.array or float
            Timestep to use for the step.
        k1 : mx.array or None, optional
            If provided, this is f(t, x) from the previous step
            (or precomputed slope). Some solvers can re-use it.
        args : dict or None, optional
            Additional arguments passed to the ODE function (if needed).

        Returns
        -------
        f_new : mx.array or None
            Slope at the new time/state, i.e. f(t+dt, x_new).
            If the solver doesn't need it (like Euler, Midpoint),
            it can be None.
        x_new : mx.array
            The new state after performing this step, i.e. x_{n+1}.
        err_est : mx.array or None
            An estimate of the local truncation error (for adaptive methods).
            If the solver is fixed-step or doesn't compute local errors,
            it can be None.
        stages : tuple of mx.array or None
            The intermediate stage slopes (k1, k2, ...) used by the solver.
            If not needed, it can be None.
        """
        raise NotImplementedError


class Euler(SolverTemplate):
    def __init__(self, dtype=mx.float32):
        """Explicit Euler ODE stepper, order 1"""
        super().__init__(order=1)
        self.dtype = dtype
        self.stepping_class = "fixed"

    def step(self, f, x, t, dt, k1=None, args=None):
        if k1 is None:
            k1 = f(t, x)
        x_sol = x + dt * k1
        return None, x_sol, None, None

    def __repr__(self):
        return "Euler"


class Midpoint(SolverTemplate):
    def __init__(self, dtype=mx.float32):
        """Explicit Midpoint (RK2) ODE stepper, order 2"""
        super().__init__(order=2)
        self.dtype = dtype
        self.stepping_class = "fixed"

    def step(self, f, x, t, dt, k1=None, args=None):
        if k1 is None:
            k1 = f(t, x)

        x_mid = x + 0.5 * dt * k1
        k2 = f(t + 0.5 * dt, x_mid)
        x_sol = x + dt * k2

        stages = (k1, k2)
        return None, x_sol, None, stages

    def __repr__(self):
        return "Midpoint"


def construct_dopri5(dtype):
    c = mx.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0], dtype=dtype)
    a = [
        mx.array([1 / 5], dtype=dtype),
        mx.array([3 / 40, 9 / 40], dtype=dtype),
        mx.array([44 / 45, -56 / 15, 32 / 9], dtype=dtype),
        mx.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], dtype=dtype),
        mx.array(
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], dtype=dtype
        ),
        mx.array(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=dtype
        ),
    ]
    bsol = mx.array(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=dtype
    )
    berr = mx.array(
        [
            1951 / 21600,
            0,
            22642 / 50085,
            451 / 720,
            -12231 / 42400,
            649 / 6300,
            1 / 60.0,
        ],
        dtype=dtype,
    )

    # midpoint evaluation not needed for dopri5
    dmid = mx.array(
        [
            -1.1270175653862835,
            0.0,
            2.675424484351598,
            -5.685526961588504,
            3.5219323679207912,
            -1.7672812570757455,
            2.382468931778144,
        ]
    )
    return (c, a, bsol, bsol - berr)


class DormandPrince45(SolverTemplate):
    def __init__(self, dtype=mx.float32):
        """Dormand–Prince (RKDP) method"""
        super().__init__(order=5)
        self.dtype = dtype
        self.stepping_class = "adaptive"
        self.tableau = construct_dopri5(self.dtype)

    def step(self, f, x, t, dt, k1=None, args=None) -> Tuple:
        if k1 is None:
            k1 = f(t, x)
        c, a, bsol, berr = self.tableau
        k2 = f(t + c[0] * dt, x + dt * a[0] * k1)
        k3 = f(t + c[1] * dt, x + dt * (a[1][0] * k1 + a[1][1] * k2))
        k4 = f(
            t + c[2] * dt, x + dt * a[2][0] * k1 + dt * a[2][1] * k2 + dt * a[2][2] * k3
        )
        k5 = f(
            t + c[3] * dt,
            x
            + dt * a[3][0] * k1
            + dt * a[3][1] * k2
            + dt * a[3][2] * k3
            + dt * a[3][3] * k4,
        )
        k6 = f(
            t + c[4] * dt,
            x
            + dt * a[4][0] * k1
            + dt * a[4][1] * k2
            + dt * a[4][2] * k3
            + dt * a[4][3] * k4
            + dt * a[4][4] * k5,
        )
        k7 = f(
            t + c[5] * dt,
            x
            + dt * a[5][0] * k1
            + dt * a[5][1] * k2
            + dt * a[5][2] * k3
            + dt * a[5][3] * k4
            + dt * a[5][4] * k5
            + dt * a[5][5] * k6,
        )
        x_sol = x + dt * (
            bsol[0] * k1
            + bsol[1] * k2
            + bsol[2] * k3
            + bsol[3] * k4
            + bsol[4] * k5
            + bsol[5] * k6
        )
        err = dt * (
            berr[0] * k1
            + berr[1] * k2
            + berr[2] * k3
            + berr[3] * k4
            + berr[4] * k5
            + berr[5] * k6
            + berr[6] * k7
        )
        return k7, x_sol, err, (k1, k2, k3, k4, k5, k6, k7)

    def __repr__(self):
        return "DormandPrince45"


SOLVER_DICT = {
    "dopri5": DormandPrince45,
    "euler": Euler,
    "midpoint": Midpoint,
}


def str_to_solver(solver_name, dtype=mx.float32):
    if solver_name not in SOLVER_DICT:
        raise ValueError(f"Invalid solver: {solver_name}")
    solver = SOLVER_DICT[solver_name]
    return solver(dtype)

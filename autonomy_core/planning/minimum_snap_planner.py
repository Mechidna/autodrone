import numpy as np


class GateTrajectoryPlanner:
    """
    Single-segment 7th-order polynomial planner.

    This is a clean drop-in replacement for a planner interface like:

        planner.update(pos0, vel0, pos1, vel1, T)
        p_ref, v_ref, a_ref = planner.sample(t)

    It uses a 7th-order polynomial in each axis:
        p(t) = c0 + c1 t + c2 t^2 + ... + c7 t^7

    Boundary conditions per axis:
        p(0)   = p0
        p'(0)  = v0
        p''(0) = a0
        p'''(0)= j0
        p(T)   = p1
        p'(T)  = v1
        p''(T) = a1
        p'''(T)= j1

    By default, a0=a1=j0=j1=0, which is a common practical choice for
    a single-segment minimum-snap-style trajectory.

    Notes:
    - For a single segment with all 8 boundary conditions fixed, this is
      effectively solving the unique 7th-order polynomial consistent with
      those constraints.
    - The true "minimum snap" optimization really becomes more interesting
      in the multi-segment case, where internal derivatives are optimized.
    """

    def __init__(self):
        self.T = 1.0
        self.coeffs = np.zeros((3, 8), dtype=float)  # one row per axis
        self.p0 = np.zeros(3, dtype=float)
        self.v0 = np.zeros(3, dtype=float)
        self.p1 = np.zeros(3, dtype=float)
        self.v1 = np.zeros(3, dtype=float)
        self.a0 = np.zeros(3, dtype=float)
        self.a1 = np.zeros(3, dtype=float)
        self.j0 = np.zeros(3, dtype=float)
        self.j1 = np.zeros(3, dtype=float)

    def update(
        self,
        p0,
        v0,
        p1,
        v1,
        T,
        a0=None,
        a1=None,
        j0=None,
        j1=None,
    ):
        """
        Build a new trajectory segment.

        Parameters
        ----------
        p0 : array-like, shape (3,)
            Initial position.
        v0 : array-like, shape (3,)
            Initial velocity.
        p1 : array-like, shape (3,)
            Final position.
        v1 : array-like, shape (3,)
            Final velocity.
        T : float
            Segment duration in seconds.
        a0, a1 : array-like, shape (3,), optional
            Initial/final acceleration. Defaults to zeros.
        j0, j1 : array-like, shape (3,), optional
            Initial/final jerk. Defaults to zeros.
        """
        self.p0 = np.asarray(p0, dtype=float).reshape(3)
        self.v0 = np.asarray(v0, dtype=float).reshape(3)
        self.p1 = np.asarray(p1, dtype=float).reshape(3)
        self.v1 = np.asarray(v1, dtype=float).reshape(3)

        self.a0 = np.zeros(3, dtype=float) if a0 is None else np.asarray(a0, dtype=float).reshape(3)
        self.a1 = np.zeros(3, dtype=float) if a1 is None else np.asarray(a1, dtype=float).reshape(3)
        self.j0 = np.zeros(3, dtype=float) if j0 is None else np.asarray(j0, dtype=float).reshape(3)
        self.j1 = np.zeros(3, dtype=float) if j1 is None else np.asarray(j1, dtype=float).reshape(3)

        self.T = max(float(T), 1e-3)

        # Solve each axis independently.
        for axis in range(3):
            self.coeffs[axis, :] = self._solve_axis(
                p0=self.p0[axis],
                v0=self.v0[axis],
                a0=self.a0[axis],
                j0=self.j0[axis],
                p1=self.p1[axis],
                v1=self.v1[axis],
                a1=self.a1[axis],
                j1=self.j1[axis],
                T=self.T,
            )

    def sample(self, t):
        """
        Sample trajectory at time t.

        Parameters
        ----------
        t : float
            Time since trajectory start.

        Returns
        -------
        p : np.ndarray, shape (3,)
            Position.
        v : np.ndarray, shape (3,)
            Velocity.
        a : np.ndarray, shape (3,)
            Acceleration.
        """
        tt = float(np.clip(t, 0.0, self.T))

        p = np.zeros(3, dtype=float)
        v = np.zeros(3, dtype=float)
        a = np.zeros(3, dtype=float)

        for axis in range(3):
            c = self.coeffs[axis, :]
            p[axis] = self._eval_poly(c, tt, order=0)
            v[axis] = self._eval_poly(c, tt, order=1)
            a[axis] = self._eval_poly(c, tt, order=2)

        return p, v, a

    def sample_full(self, t):
        """
        Optional helper if you later want jerk too.
        """
        tt = float(np.clip(t, 0.0, self.T))

        p = np.zeros(3, dtype=float)
        v = np.zeros(3, dtype=float)
        a = np.zeros(3, dtype=float)
        j = np.zeros(3, dtype=float)

        for axis in range(3):
            c = self.coeffs[axis, :]
            p[axis] = self._eval_poly(c, tt, order=0)
            v[axis] = self._eval_poly(c, tt, order=1)
            a[axis] = self._eval_poly(c, tt, order=2)
            j[axis] = self._eval_poly(c, tt, order=3)

        return p, v, a, j

    def _solve_axis(self, p0, v0, a0, j0, p1, v1, a1, j1, T):
        """
        Solve one axis of the 7th-order polynomial.
        """
        A = np.array([
            # t = 0 constraints
            [1, 0, 0, 0, 0, 0, 0, 0],                    # p(0)
            [0, 1, 0, 0, 0, 0, 0, 0],                    # v(0)
            [0, 0, 2, 0, 0, 0, 0, 0],                    # a(0)
            [0, 0, 0, 6, 0, 0, 0, 0],                    # j(0)

            # t = T constraints
            [1, T, T**2, T**3, T**4, T**5, T**6, T**7],  # p(T)
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4, 6*T**5, 7*T**6],  # v(T)
            [0, 0, 2, 6*T, 12*T**2, 20*T**3, 30*T**4, 42*T**5],   # a(T)
            [0, 0, 0, 6, 24*T, 60*T**2, 120*T**3, 210*T**4],      # j(T)
        ], dtype=float)

        b = np.array([p0, v0, a0, j0, p1, v1, a1, j1], dtype=float)

        coeffs = np.linalg.solve(A, b)
        return coeffs

    @staticmethod
    def _eval_poly(c, t, order=0):
        """
        Evaluate polynomial or its derivatives.

        c: coefficients [c0..c7]
        order:
            0 -> position
            1 -> velocity
            2 -> acceleration
            3 -> jerk
            4 -> snap
        """
        c0, c1, c2, c3, c4, c5, c6, c7 = c

        if order == 0:
            return (
                c0
                + c1*t
                + c2*t**2
                + c3*t**3
                + c4*t**4
                + c5*t**5
                + c6*t**6
                + c7*t**7
            )

        if order == 1:
            return (
                c1
                + 2*c2*t
                + 3*c3*t**2
                + 4*c4*t**3
                + 5*c5*t**4
                + 6*c6*t**5
                + 7*c7*t**6
            )

        if order == 2:
            return (
                2*c2
                + 6*c3*t
                + 12*c4*t**2
                + 20*c5*t**3
                + 30*c6*t**4
                + 42*c7*t**5
            )

        if order == 3:
            return (
                6*c3
                + 24*c4*t
                + 60*c5*t**2
                + 120*c6*t**3
                + 210*c7*t**4
            )

        if order == 4:
            return (
                24*c4
                + 120*c5*t
                + 360*c6*t**2
                + 840*c7*t**3
            )

        raise ValueError(f"Unsupported derivative order: {order}")
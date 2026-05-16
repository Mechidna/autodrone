import math
from typing import Optional, Sequence, Tuple
from scipy.optimize import minimize

import numpy as np


class MultiSegmentMinimumSnapPlanner:
    """
    Multi-segment 7th-order minimum-snap trajectory planner.

    Solves, independently per axis:

        minimize    0.5 * c^T Q c
        subject to  A c = b

    where c contains all polynomial coefficients for all segments.

    Each segment k uses a 7th-order polynomial in local time tau in [0, T_k]:

        p_k(tau) = c0 + c1 tau + c2 tau^2 + ... + c7 tau^7

    Constraints:
    - Segment start/end positions match the supplied waypoint list.
    - Start velocity/acceleration/jerk are constrained.
    - End velocity/acceleration/jerk are constrained.
    - Internal boundaries enforce continuity of derivatives 1..6.

    Why this works:
    - For M segments, there are 8*M coefficients per axis.
    - Constraints count is exactly 8*M:
        * 2*M position constraints
        * 3 start derivative constraints (v,a,j)
        * 3 end derivative constraints   (v,a,j)
        * 6*(M-1) internal continuity constraints for derivatives 1..6

    This gives a square, fully constrained equality-constrained QP.

    Public API:
    - update(...)
    - sample(t) -> (p, v, a)
    - sample_full(t) -> (p, v, a, j, s)

    Notes:
    - `times` are per-segment durations, length M.
    - `waypoints` must have length M+1 and shape (M+1, 3).
    - Local polynomial time resets to 0 at the start of each segment.
    """

    POLY_ORDER = 7
    N_COEFF = 8
    SNAP_ORDER = 4

    def __init__(self):
        self.waypoints: Optional[np.ndarray] = None      # shape (M+1, 3)
        self.times: Optional[np.ndarray] = None          # shape (M,)
        self.coeffs: Optional[np.ndarray] = None         # shape (M, 3, 8)
        self.segment_starts: Optional[np.ndarray] = None # shape (M+1,)
        self.total_time: float = 0.0
        self.num_segments: int = 0

        # stored boundary conditions
        self.v_start = np.zeros(3, dtype=float)
        self.v_end = np.zeros(3, dtype=float)
        self.a_start = np.zeros(3, dtype=float)
        self.a_end = np.zeros(3, dtype=float)
        self.j_start = np.zeros(3, dtype=float)
        self.j_end = np.zeros(3, dtype=float)

    def update(
        self,
        waypoints: Sequence[Sequence[float]],
        times: Sequence[float],
        v_start: Optional[Sequence[float]] = None,
        v_end: Optional[Sequence[float]] = None,
        a_start: Optional[Sequence[float]] = None,
        a_end: Optional[Sequence[float]] = None,
        j_start: Optional[Sequence[float]] = None,
        j_end: Optional[Sequence[float]] = None,
        waypoint_velocities: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """
        Build a new multi-segment trajectory through all supplied waypoints.

        Parameters
        ----------
        waypoints : array-like, shape (M+1, 3)
            Waypoints the trajectory must pass through.
            Example: [start, gate1, gate2, gate3]
        times : array-like, shape (M,)
            Duration of each segment.
            Example: [T0, T1, T2] for 4 waypoints / 3 segments.
        v_start, v_end : array-like, shape (3,), optional
            Boundary velocity at the first and last waypoint.
            Defaults to zero.
        a_start, a_end : array-like, shape (3,), optional
            Boundary acceleration at the first and last waypoint.
            Defaults to zero.
        j_start, j_end : array-like, shape (3,), optional
            Boundary jerk at the first and last waypoint.
            Defaults to zero.
        """
        wp = np.asarray(waypoints, dtype=float)
        ts = np.asarray(times, dtype=float).reshape(-1)

        if wp.ndim != 2 or wp.shape[1] != 3:
            raise ValueError(f"`waypoints` must have shape (M+1, 3), got {wp.shape}")

        if ts.ndim != 1:
            raise ValueError(f"`times` must be 1D, got shape {ts.shape}")

        if len(wp) < 2:
            raise ValueError("At least 2 waypoints are required.")

        m = len(wp) - 1
        if len(ts) != m:
            raise ValueError(
                f"`times` length must be number of segments = len(waypoints)-1 = {m}, got {len(ts)}"
            )

        if np.any(ts <= 1e-6):
            raise ValueError("All segment durations must be positive.")

        self.waypoints = wp
        self.times = ts
        self.num_segments = m
        self.segment_starts = np.concatenate(([0.0], np.cumsum(ts)))
        self.total_time = float(np.sum(ts))

        self.v_start = self._vec3_or_zero(v_start)
        self.v_end = self._vec3_or_zero(v_end)
        self.a_start = self._vec3_or_zero(a_start)
        self.a_end = self._vec3_or_zero(a_end)
        self.j_start = self._vec3_or_zero(j_start)
        self.j_end = self._vec3_or_zero(j_end)

        waypoint_velocities_arr = None
        if waypoint_velocities is not None:
            waypoint_velocities_arr = np.asarray(waypoint_velocities, dtype=float)
            if waypoint_velocities_arr.shape != wp.shape:
                raise ValueError(
                    "`waypoint_velocities` must match `waypoints` shape "
                    f"{wp.shape}, got {waypoint_velocities_arr.shape}"
                )

        self.coeffs = np.zeros((m, 3, self.N_COEFF), dtype=float)

        # Solve each axis independently.
        for axis in range(3):
            coeff_axis = self._solve_axis(
                waypoints_1d=wp[:, axis],
                times=ts,
                d_start=np.array(
                    [self.v_start[axis], self.a_start[axis], self.j_start[axis]],
                    dtype=float,
                ),
                d_end=np.array(
                    [self.v_end[axis], self.a_end[axis], self.j_end[axis]],
                    dtype=float,
                ),
                waypoint_velocities=None
                if waypoint_velocities_arr is None
                else waypoint_velocities_arr[:, axis],
            )
            self.coeffs[:, axis, :] = coeff_axis

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample the trajectory at global time t.

        Parameters
        ----------
        t : float
            Time since the trajectory start.

        Returns
        -------
        p, v, a : tuple of np.ndarray, each shape (3,)
            Position, velocity, acceleration.
        """
        seg_idx, tau = self._locate_segment(t)

        p = np.zeros(3, dtype=float)
        v = np.zeros(3, dtype=float)
        a = np.zeros(3, dtype=float)

        for axis in range(3):
            c = self.coeffs[seg_idx, axis, :]
            p[axis] = self._eval_poly(c, tau, order=0)
            v[axis] = self._eval_poly(c, tau, order=1)
            a[axis] = self._eval_poly(c, tau, order=2)

        return p, v, a

    def sample_full(
        self, t: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample the trajectory at global time t and also return jerk and snap.

        Returns
        -------
        p, v, a, j, s : tuple of np.ndarray, each shape (3,)
        """
        seg_idx, tau = self._locate_segment(t)

        p = np.zeros(3, dtype=float)
        v = np.zeros(3, dtype=float)
        a = np.zeros(3, dtype=float)
        j = np.zeros(3, dtype=float)
        s = np.zeros(3, dtype=float)

        for axis in range(3):
            c = self.coeffs[seg_idx, axis, :]
            p[axis] = self._eval_poly(c, tau, order=0)
            v[axis] = self._eval_poly(c, tau, order=1)
            a[axis] = self._eval_poly(c, tau, order=2)
            j[axis] = self._eval_poly(c, tau, order=3)
            s[axis] = self._eval_poly(c, tau, order=4)

        return p, v, a, j, s

    def get_segment_endpoint_state(
        self, seg_idx: int, at_end: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience helper for debugging segment transitions.

        Parameters
        ----------
        seg_idx : int
            Segment index.
        at_end : bool
            If True, evaluate at tau = T_seg, otherwise tau = 0.

        Returns
        -------
        p, v, a : np.ndarray
        """
        if self.coeffs is None:
            raise RuntimeError("Planner has not been updated yet.")

        if not (0 <= seg_idx < self.num_segments):
            raise IndexError(f"Segment index out of range: {seg_idx}")

        tau = self.times[seg_idx] if at_end else 0.0

        p = np.zeros(3, dtype=float)
        v = np.zeros(3, dtype=float)
        a = np.zeros(3, dtype=float)

        for axis in range(3):
            c = self.coeffs[seg_idx, axis, :]
            p[axis] = self._eval_poly(c, tau, order=0)
            v[axis] = self._eval_poly(c, tau, order=1)
            a[axis] = self._eval_poly(c, tau, order=2)

        return p, v, a

    # -------------------------------------------------------------------------
    # Core solve
    # -------------------------------------------------------------------------

    def _solve_axis(
        self,
        waypoints_1d: np.ndarray,
        times: np.ndarray,
        d_start: np.ndarray,  # [v0, a0, j0]
        d_end: np.ndarray,    # [vT, aT, jT]
        waypoint_velocities: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve one axis for all segments at once.

        Returns
        -------
        coeffs : np.ndarray, shape (M, 8)
        """
        m = len(times)
        n = self.N_COEFF * m

        Q = self._build_global_Q(times)
        A, b = self._build_constraints(
            waypoints_1d,
            times,
            d_start,
            d_end,
            waypoint_velocities=waypoint_velocities,
        )

        if A.shape != (n, n):
            raise RuntimeError(
                f"Constraint matrix should be square ({n}, {n}), got {A.shape}"
            )
        if b.shape != (n,):
            raise RuntimeError(f"Constraint vector should have shape ({n},), got {b.shape}")

        # Symmetrize Q to reduce tiny numerical asymmetries.
        Q = 0.5 * (Q + Q.T)

        # Small regularization can help if times are extreme.
        reg = 1e-10
        Q_reg = Q + reg * np.eye(n)

        # KKT system:
        # [Q  A^T] [c] = [0]
        # [A   0 ] [λ]   [b]
        KKT = np.zeros((n + n, n + n), dtype=float)
        rhs = np.zeros(n + n, dtype=float)

        KKT[:n, :n] = Q_reg
        KKT[:n, n:] = A.T
        KKT[n:, :n] = A
        rhs[n:] = b

        sol = np.linalg.solve(KKT, rhs)
        c = sol[:n]

        coeffs = c.reshape(m, self.N_COEFF)
        return coeffs

    def _build_global_Q(self, times: np.ndarray) -> np.ndarray:
        """
        Build block-diagonal minimum-snap cost matrix over all segments.
        """
        m = len(times)
        n = self.N_COEFF * m
        Q = np.zeros((n, n), dtype=float)

        for k, T in enumerate(times):
            Qk = self._segment_Q(T)
            sl = slice(k * self.N_COEFF, (k + 1) * self.N_COEFF)
            Q[sl, sl] = Qk

        return Q

    def _build_constraints(
        self,
        waypoints_1d: np.ndarray,
        times: np.ndarray,
        d_start: np.ndarray,
        d_end: np.ndarray,
        waypoint_velocities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build equality constraints A c = b for one axis.
        """
        m = len(times)
        n = self.N_COEFF * m

        rows = []
        vals = []

        def add_row(row: np.ndarray, value: float) -> None:
            rows.append(row)
            vals.append(float(value))

        # 1) Position constraints for each segment start/end:
        #    p_k(0)   = waypoint[k]
        #    p_k(T_k) = waypoint[k+1]
        for k, T in enumerate(times):
            row_start = np.zeros(n, dtype=float)
            row_end = np.zeros(n, dtype=float)

            sl = slice(k * self.N_COEFF, (k + 1) * self.N_COEFF)
            row_start[sl] = self._basis_row(t=0.0, order=0)
            row_end[sl] = self._basis_row(t=T, order=0)

            add_row(row_start, waypoints_1d[k])
            add_row(row_end, waypoints_1d[k + 1])

        # 2) Start derivative constraints: v, a, j at segment 0 start
        for deriv_order, target in zip((1, 2, 3), d_start):
            row = np.zeros(n, dtype=float)
            row[0:self.N_COEFF] = self._basis_row(t=0.0, order=deriv_order)
            add_row(row, target)

        # 3) End derivative constraints: v, a, j at final segment end
        last_sl = slice((m - 1) * self.N_COEFF, m * self.N_COEFF)
        T_last = times[-1]
        for deriv_order, target in zip((1, 2, 3), d_end):
            row = np.zeros(n, dtype=float)
            row[last_sl] = self._basis_row(t=T_last, order=deriv_order)
            add_row(row, target)

        constrained_internal_velocity = set()
        if waypoint_velocities is not None:
            waypoint_velocities = np.asarray(waypoint_velocities, dtype=float).reshape(-1)
            if len(waypoint_velocities) != len(waypoints_1d):
                raise ValueError(
                    "`waypoint_velocities` length must match waypoints length "
                    f"{len(waypoints_1d)}, got {len(waypoint_velocities)}"
                )
            constrained_internal_velocity = {
                i for i in range(1, len(waypoint_velocities) - 1)
                if np.isfinite(waypoint_velocities[i])
            }

        # 4) Internal continuity constraints for derivatives 1..6.
        # If an internal waypoint velocity is specified, replace derivative-1
        # continuity and derivative-6 continuity at that boundary with explicit
        # end/start velocity constraints. This keeps the square 8M system while
        # preserving position and lower-order smoothness through acceleration,
        # jerk, snap, and crackle.
        #    d^r/dt^r p_k(T_k) = d^r/dt^r p_{k+1}(0)
        for k in range(m - 1):
            T = times[k]
            sl_k = slice(k * self.N_COEFF, (k + 1) * self.N_COEFF)
            sl_k1 = slice((k + 1) * self.N_COEFF, (k + 2) * self.N_COEFF)
            waypoint_idx = k + 1

            if waypoint_idx in constrained_internal_velocity:
                target_v = float(waypoint_velocities[waypoint_idx])

                row_end_v = np.zeros(n, dtype=float)
                row_end_v[sl_k] = self._basis_row(t=T, order=1)
                add_row(row_end_v, target_v)

                row_start_v = np.zeros(n, dtype=float)
                row_start_v[sl_k1] = self._basis_row(t=0.0, order=1)
                add_row(row_start_v, target_v)

            for deriv_order in range(1, 7):
                if waypoint_idx in constrained_internal_velocity and deriv_order in (1, 6):
                    continue
                row = np.zeros(n, dtype=float)
                row[sl_k] = self._basis_row(t=T, order=deriv_order)
                row[sl_k1] = -self._basis_row(t=0.0, order=deriv_order)
                add_row(row, 0.0)

        A = np.vstack(rows)
        b = np.asarray(vals, dtype=float)

        return A, b

    # -------------------------------------------------------------------------
    # Math helpers
    # -------------------------------------------------------------------------

    @classmethod
    def _segment_Q(cls, T: float) -> np.ndarray:
        """
        Cost matrix for one segment:

            J = integral_0^T [p''''(t)]^2 dt

        For p(t) = sum_i c_i t^i, only coefficients i >= 4 contribute.
        """
        Q = np.zeros((cls.N_COEFF, cls.N_COEFF), dtype=float)

        for i in range(cls.N_COEFF):
            for j in range(cls.N_COEFF):
                if i < cls.SNAP_ORDER or j < cls.SNAP_ORDER:
                    continue

                coef_i = cls._falling_factorial(i, cls.SNAP_ORDER)
                coef_j = cls._falling_factorial(j, cls.SNAP_ORDER)

                power = i + j - 2 * cls.SNAP_ORDER
                # integral t^power dt from 0 to T = T^(power+1)/(power+1)
                Q[i, j] = coef_i * coef_j * (T ** (power + 1)) / (power + 1)

        return Q

    @classmethod
    def _basis_row(cls, t: float, order: int) -> np.ndarray:
        """
        Basis row such that:

            basis_row(t, order) @ c = p^(order)(t)

        for c = [c0, c1, ..., c7].
        """
        row = np.zeros(cls.N_COEFF, dtype=float)

        for i in range(order, cls.N_COEFF):
            row[i] = cls._falling_factorial(i, order) * (t ** (i - order))

        return row

    @staticmethod
    def _falling_factorial(n: int, k: int) -> float:
        """
        n * (n-1) * ... * (n-k+1), with convention result = 1 for k=0.
        """
        if k < 0:
            raise ValueError("k must be >= 0")
        if k == 0:
            return 1.0
        if n < k:
            return 0.0

        out = 1.0
        for x in range(k):
            out *= (n - x)
        return out

    @classmethod
    def _eval_poly(cls, c: np.ndarray, t: float, order: int = 0) -> float:
        """
        Evaluate polynomial derivative of requested order.
        """
        return float(cls._basis_row(t, order) @ c)

    # -------------------------------------------------------------------------
    # Trajectory-time helpers
    # -------------------------------------------------------------------------

    def _locate_segment(self, t: float) -> Tuple[int, float]:
        """
        Convert global time t into (segment_index, local_tau).
        """
        if self.coeffs is None or self.times is None or self.segment_starts is None:
            raise RuntimeError("Planner has not been updated yet.")

        tt = float(np.clip(t, 0.0, self.total_time))

        # Special-case final endpoint.
        if tt >= self.total_time:
            seg_idx = self.num_segments - 1
            tau = float(self.times[seg_idx])
            return seg_idx, tau

        # segment_starts = [0, T0, T0+T1, ...]
        seg_idx = int(np.searchsorted(self.segment_starts, tt, side="right") - 1)
        seg_idx = max(0, min(seg_idx, self.num_segments - 1))
        tau = tt - self.segment_starts[seg_idx]
        return seg_idx, float(tau)

    @staticmethod
    def _vec3_or_zero(x: Optional[Sequence[float]]) -> np.ndarray:
        if x is None:
            return np.zeros(3, dtype=float)
        arr = np.asarray(x, dtype=float).reshape(3)
        return arr

    def compute_total_snap_cost(self) -> float:
        """
        Compute total integrated squared snap cost for the currently solved trajectory.

        Returns
        -------
        float
            Sum over all segments and all 3 axes of c^T Q c.
        """
        if self.coeffs is None or self.times is None:
            raise RuntimeError("Planner has not been updated yet.")

        total = 0.0
        for k, T in enumerate(self.times):
            Qk = self._segment_Q(float(T))
            for axis in range(3):
                c = self.coeffs[k, axis, :]
                total += float(c @ Qk @ c)

        return total

    def optimize_times(
            self,
            waypoints,
            times_init,
            v_start=None,
            v_end=None,
            a_start=None,
            a_end=None,
            j_start=None,
            j_end=None,
            lambda_time: float = 1.0,
            lambda_snap: float = 1.0,
            t_min: float = 0.05,
            maxiter: int = 100,
    ):
        """
        Optimize segment durations using an outer optimization loop.

        Objective:
            F(T) = lambda_time * sum(T) + lambda_snap * snap_cost(T)

        where snap_cost(T) is computed after solving the minimum-snap coefficients
        for the candidate segment times T.

        Parameters
        ----------
        waypoints : array-like, shape (M+1, 3)
        times_init : array-like, shape (M,)
            Initial guess for segment durations.
        lambda_time : float
            Weight on total time. Larger -> faster trajectories preferred.
        lambda_snap : float
            Weight on smoothness. Larger -> smoother/slower trajectories preferred.
        t_min : float
            Minimum allowed segment time.
        maxiter : int
            Maximum iterations for the outer optimizer.

        Returns
        -------
        times_opt : np.ndarray
            Optimized segment durations.
        result : OptimizeResult
            Raw SciPy optimization result.
        """
        wp = np.asarray(waypoints, dtype=float)
        times_init = np.asarray(times_init, dtype=float).reshape(-1)

        if np.any(times_init <= t_min):
            raise ValueError(f"All initial times must be > t_min={t_min}")

        # Optimize unconstrained variables u, map to positive times with:
        # T = t_min + exp(u)
        def unpack_times(u: np.ndarray) -> np.ndarray:
            return t_min + np.exp(u)

        def objective(u: np.ndarray) -> float:
            times = unpack_times(u)

            self.update(
                waypoints=wp,
                times=times,
                v_start=v_start,
                v_end=v_end,
                a_start=a_start,
                a_end=a_end,
                j_start=j_start,
                j_end=j_end,
            )

            snap_cost = self.compute_total_snap_cost()
            total_time = float(np.sum(times))

            return lambda_time * total_time + lambda_snap * snap_cost

        u0 = np.log(times_init - t_min)

        result = minimize(
            objective,
            u0,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        )

        times_opt = unpack_times(result.x)

        # Rebuild planner using optimal times
        self.update(
            waypoints=wp,
            times=times_opt,
            v_start=v_start,
            v_end=v_end,
            a_start=a_start,
            a_end=a_end,
            j_start=j_start,
            j_end=j_end,
        )

        return times_opt, result

if __name__ == "__main__":
    planner = MultiSegmentMinimumSnapPlanner()

    waypoints = [
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        [4.0, 0.0, 2.0],
        [6.0, 2.0, 1.5],
    ]

    times_init = [1.5, 2.0, 1.8]

    times_opt, result = planner.optimize_times(
        waypoints=waypoints,
        times_init=times_init,
        v_start=[0, 0, 0],
        v_end=[0, 0, 0],
        a_start=[0, 0, 0],
        a_end=[0, 0, 0],
        j_start=[0, 0, 0],
        j_end=[0, 0, 0],
        lambda_time=1.0,
        lambda_snap=0.01,
        t_min=0.1,
        maxiter=100,
    )

    print("Optimized times:", times_opt)
    print("Optimization success:", result.success)
    print("Final objective:", result.fun)

    p, v, a, j, s = planner.sample_full(2.3)
    print("Sample at t=2.3")
    print("p =", p)
    print("v =", v)
    print("a =", a)
    print("j =", j)
    print("s =", s)

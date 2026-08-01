import copy
import math
from typing import Optional, Sequence, Tuple
from scipy.linalg import block_diag, null_space
from scipy.optimize import linprog, minimize

import numpy as np


class MultiSegmentMinimumSnapPlanner:
    """
    Multi-segment 7th-order minimum-snap trajectory planner.

    By default, solves independently per axis:

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

    When ``forward_progress_enabled`` is requested, the planner instead solves
    one joint 3-D minimum-snap problem.  Acceleration and jerk remain continuous
    at internal waypoints, but the unnecessary crackle/pop constraints are
    released so the snap objective has degrees of freedom.  Linear Bernstein
    constraints require the velocity projected onto every waypoint chord to
    remain nonnegative over the *entire* segment.  If that sufficient condition
    is overly conservative, the solver switches to sampled constraints and
    adds the exact negative polynomial extrema back into the QP until none
    remain.  This prevents the high-order interpolation lobes which can
    otherwise make a trajectory turn around between two forward-ordered gates.

    Public API:
    - update(...)
    - retimed(scale) -> independent trajectory with identical spatial path
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

        # Diagnostics for the optional joint constrained solve.
        self.forward_progress_enabled = False
        self.forward_progress_min_speed_m_s = 0.0
        self.forward_progress_solver_status = "disabled"
        self.forward_progress_min_bernstein_speed_m_s = float("nan")
        self.forward_progress_min_speed_m_s_solved = float("nan")

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
        forward_progress_enabled: bool = False,
        forward_progress_min_speed_m_s: float = 0.0,
        forward_progress_solver_max_iterations: int = 300,
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
        forward_progress_enabled : bool, optional
            Solve a joint constrained minimum-snap problem that prohibits
            velocity reversal along each ordered waypoint segment.
        forward_progress_min_speed_m_s : float, optional
            Lower bound on local chord-projected velocity.  Zero prohibits
            reversal while still allowing a stationary endpoint.
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
        self.forward_progress_enabled = bool(forward_progress_enabled)
        self.forward_progress_min_speed_m_s = max(
            0.0,
            float(forward_progress_min_speed_m_s),
        )
        self.forward_progress_solver_status = "disabled"
        self.forward_progress_min_bernstein_speed_m_s = float("nan")
        self.forward_progress_min_speed_m_s_solved = float("nan")

        waypoint_velocities_arr = None
        if waypoint_velocities is not None:
            waypoint_velocities_arr = np.asarray(waypoint_velocities, dtype=float)
            if waypoint_velocities_arr.shape != wp.shape:
                raise ValueError(
                    "`waypoint_velocities` must match `waypoints` shape "
                    f"{wp.shape}, got {waypoint_velocities_arr.shape}"
                )

        if self.forward_progress_enabled:
            self.coeffs = self._solve_forward_constrained(
                waypoints=wp,
                times=ts,
                waypoint_velocities=waypoint_velocities_arr,
                min_forward_speed_m_s=self.forward_progress_min_speed_m_s,
                max_iterations=max(1, int(forward_progress_solver_max_iterations)),
            )
        else:
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

    def retimed(self, scale: float) -> "MultiSegmentMinimumSnapPlanner":
        """Return an independent, uniformly time-dilated trajectory.

        This operation does not run the optimizer again.  For a scale ``s``,
        the returned trajectory satisfies ``p_new(t) = p_old(t / s)``.  Its
        spatial curve is therefore identical, while velocity, acceleration,
        jerk, and snap are divided by ``s``, ``s**2``, ``s**3``, and ``s**4``
        respectively.
        """
        if self.coeffs is None or self.times is None or self.waypoints is None:
            raise RuntimeError("Planner has not been updated yet.")
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"`scale` must be finite and positive, got {scale}")

        retimed = copy.deepcopy(self)
        retimed.times = np.asarray(self.times, dtype=float) * scale
        retimed.segment_starts = np.concatenate(
            ([0.0], np.cumsum(retimed.times))
        )
        retimed.total_time = float(np.sum(retimed.times))

        powers = scale ** np.arange(self.N_COEFF, dtype=float)
        retimed.coeffs = np.asarray(self.coeffs, dtype=float) / powers[None, None, :]
        retimed.v_start = np.asarray(self.v_start, dtype=float) / scale
        retimed.v_end = np.asarray(self.v_end, dtype=float) / scale
        retimed.a_start = np.asarray(self.a_start, dtype=float) / (scale ** 2)
        retimed.a_end = np.asarray(self.a_end, dtype=float) / (scale ** 2)
        retimed.j_start = np.asarray(self.j_start, dtype=float) / (scale ** 3)
        retimed.j_end = np.asarray(self.j_end, dtype=float) / (scale ** 3)

        if math.isfinite(retimed.forward_progress_min_speed_m_s):
            retimed.forward_progress_min_speed_m_s /= scale
        if math.isfinite(retimed.forward_progress_min_bernstein_speed_m_s):
            retimed.forward_progress_min_bernstein_speed_m_s /= scale
        if math.isfinite(retimed.forward_progress_min_speed_m_s_solved):
            retimed.forward_progress_min_speed_m_s_solved /= scale
        retimed.forward_progress_solver_status = (
            f"{retimed.forward_progress_solver_status}:retimed={scale:.6g}"
        )

        # Preserve the captured raw vehicle velocity for diagnostics, but make
        # the reported boundary velocity agree with the actual retimed curve.
        if hasattr(retimed, "_aigp_v_start_used"):
            retimed._aigp_v_start_used = (
                np.asarray(retimed._aigp_v_start_used, dtype=float) / scale
            )
        return retimed

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

    def _solve_forward_constrained(
        self,
        *,
        waypoints: np.ndarray,
        times: np.ndarray,
        waypoint_velocities: Optional[np.ndarray],
        min_forward_speed_m_s: float,
        max_iterations: int,
    ) -> np.ndarray:
        """Solve one joint 3-D QP with segment-local forward inequalities.

        Coefficients are solved in normalized segment time ``u in [0, 1]`` to
        keep the equality and cost matrices well conditioned.  The result is
        converted back to the class' local-time coefficient convention.
        """
        m = len(times)
        n_axis = self.N_COEFF * m
        q_axis = self._build_normalized_global_Q(times)

        particular = []
        nullspaces = []
        reduced_hessians = []
        reduced_gradients = []

        for axis in range(3):
            velocities_1d = (
                None
                if waypoint_velocities is None
                else waypoint_velocities[:, axis]
            )
            a_eq, b_eq = self._build_forward_equality_constraints(
                waypoints_1d=waypoints[:, axis],
                times=times,
                d_start=np.array(
                    [self.v_start[axis], self.a_start[axis], self.j_start[axis]],
                    dtype=float,
                ),
                d_end=np.array(
                    [self.v_end[axis], self.a_end[axis], self.j_end[axis]],
                    dtype=float,
                ),
                waypoint_velocities=velocities_1d,
            )
            c0, _, rank, _ = np.linalg.lstsq(a_eq, b_eq, rcond=1e-11)
            residual = float(np.max(np.abs(a_eq @ c0 - b_eq)))
            if residual > 1e-6:
                raise RuntimeError(
                    "Forward-progress equality constraints are inconsistent: "
                    f"axis={axis}, residual={residual:.3e}, rank={rank}"
                )
            z_axis = null_space(a_eq, rcond=1e-11)
            particular.append(c0)
            nullspaces.append(z_axis)
            reduced_hessians.append(z_axis.T @ q_axis @ z_axis)
            reduced_gradients.append(z_axis.T @ q_axis @ c0)

        free_sizes = [z.shape[1] for z in nullspaces]
        offsets = np.cumsum([0] + free_sizes)
        total_free = int(offsets[-1])
        if total_free > 0:
            hessian = block_diag(*reduced_hessians)
            gradient = np.concatenate(reduced_gradients)
        else:
            hessian = np.zeros((0, 0), dtype=float)
            gradient = np.zeros(0, dtype=float)

        inequality_rows = []
        inequality_rhs = []
        inequality_constants = []
        for seg_idx, duration in enumerate(times):
            chord = waypoints[seg_idx + 1] - waypoints[seg_idx]
            chord_length = float(np.linalg.norm(chord))
            if not math.isfinite(chord_length) or chord_length < 1e-6:
                continue
            direction = chord / chord_length
            for bernstein_idx in range(self.POLY_ORDER):
                basis = self._velocity_bernstein_row(
                    bernstein_idx,
                    float(duration),
                )
                row = np.zeros(total_free, dtype=float)
                constant = 0.0
                segment_slice = slice(
                    seg_idx * self.N_COEFF,
                    (seg_idx + 1) * self.N_COEFF,
                )
                for axis in range(3):
                    weighted_basis = float(direction[axis]) * basis
                    constant += float(
                        weighted_basis @ particular[axis][segment_slice]
                    )
                    if free_sizes[axis] > 0:
                        row[offsets[axis]:offsets[axis + 1]] = (
                            weighted_basis @ nullspaces[axis][segment_slice, :]
                        )
                inequality_rows.append(row)
                inequality_rhs.append(float(min_forward_speed_m_s) - constant)
                inequality_constants.append(constant)

        b_ineq = (
            np.vstack(inequality_rows)
            if inequality_rows
            else np.zeros((0, total_free), dtype=float)
        )
        rhs_ineq = np.asarray(inequality_rhs, dtype=float)
        constants = np.asarray(inequality_constants, dtype=float)
        constraint_mode = "bernstein"

        if total_free == 0:
            min_value = float(np.min(constants)) if len(constants) else float("inf")
            if min_value < float(min_forward_speed_m_s) - 1e-7:
                raise RuntimeError(
                    "Forward-progress constraints are infeasible with the fixed "
                    f"boundary conditions (minimum={min_value:.3f} m/s)."
                )
            solution_reduced = np.zeros(0, dtype=float)
            self.forward_progress_solver_status = "fixed_feasible"
        else:
            hessian = 0.5 * (hessian + hessian.T)
            objective_scale = max(
                1.0,
                float(np.max(np.abs(hessian))) if hessian.size else 0.0,
                float(np.max(np.abs(gradient))) if gradient.size else 0.0,
            )
            h_scaled = hessian / objective_scale
            g_scaled = gradient / objective_scale
            regularization = 1e-10
            h_regularized = h_scaled + regularization * np.eye(total_free)
            try:
                initial = -np.linalg.solve(h_regularized, g_scaled)
            except np.linalg.LinAlgError:
                initial = -np.linalg.lstsq(
                    h_regularized,
                    g_scaled,
                    rcond=1e-12,
                )[0]

            initial_margin = b_ineq @ initial - rhs_ineq
            if len(initial_margin) == 0 or float(np.min(initial_margin)) >= -1e-8:
                solution_reduced = initial
                self.forward_progress_solver_status = "unconstrained_feasible"
            else:
                # SLSQP is much more reliable when initialized inside the
                # feasible polytope.  HiGHS supplies a feasible point for the
                # linear Bernstein inequalities without changing the QP's
                # objective.
                feasibility = linprog(
                    c=np.zeros(total_free, dtype=float),
                    A_ub=-b_ineq,
                    b_ub=-rhs_ineq,
                    bounds=[(None, None)] * total_free,
                    method="highs",
                )
                if not feasibility.success:
                    # Nonnegative Bernstein control points are sufficient but
                    # not necessary for a nonnegative polynomial.  A short
                    # exit segment that brakes from pass-through speed to zero
                    # is a common valid case that can fail that conservative
                    # test.  Fall back to dense linear velocity constraints;
                    # the solved polynomial is checked at all exact extrema
                    # below.
                    constraint_mode = "extrema"
                    inequality_rows = []
                    inequality_rhs = []
                    inequality_constants = []
                    for seg_idx, duration in enumerate(times):
                        chord = waypoints[seg_idx + 1] - waypoints[seg_idx]
                        chord_length = float(np.linalg.norm(chord))
                        if not math.isfinite(chord_length) or chord_length < 1e-6:
                            continue
                        direction = chord / chord_length
                        segment_slice = slice(
                            seg_idx * self.N_COEFF,
                            (seg_idx + 1) * self.N_COEFF,
                        )
                        for normalized_time in np.linspace(0.0, 1.0, 65):
                            basis = self._basis_row(
                                float(normalized_time),
                                1,
                            ) / float(duration)
                            row = np.zeros(total_free, dtype=float)
                            constant = 0.0
                            for axis in range(3):
                                weighted_basis = float(direction[axis]) * basis
                                constant += float(
                                    weighted_basis
                                    @ particular[axis][segment_slice]
                                )
                                if free_sizes[axis] > 0:
                                    row[offsets[axis]:offsets[axis + 1]] = (
                                        weighted_basis
                                        @ nullspaces[axis][segment_slice, :]
                                    )
                            inequality_rows.append(row)
                            inequality_rhs.append(
                                float(min_forward_speed_m_s) - constant
                            )
                            inequality_constants.append(constant)
                    b_ineq = np.vstack(inequality_rows)
                    rhs_ineq = np.asarray(inequality_rhs, dtype=float)
                    constants = np.asarray(inequality_constants, dtype=float)
                    feasibility = linprog(
                        c=np.zeros(total_free, dtype=float),
                        A_ub=-b_ineq,
                        b_ub=-rhs_ineq,
                        bounds=[(None, None)] * total_free,
                        method="highs",
                    )
                    if not feasibility.success:
                        raise RuntimeError(
                            "Forward-progress constraints are infeasible: "
                            f"status={feasibility.status}, "
                            f"message={feasibility.message}"
                        )
                feasible_initial = np.asarray(feasibility.x, dtype=float)

                def objective(z: np.ndarray) -> float:
                    return float(0.5 * z @ h_regularized @ z + g_scaled @ z)

                def objective_jac(z: np.ndarray) -> np.ndarray:
                    return h_regularized @ z + g_scaled

                result = minimize(
                    objective,
                    feasible_initial,
                    jac=objective_jac,
                    method="SLSQP",
                    constraints={
                        "type": "ineq",
                        "fun": lambda z: b_ineq @ z - rhs_ineq,
                        "jac": lambda _z: b_ineq,
                    },
                    options={
                        "maxiter": int(max_iterations),
                        "ftol": 1e-10,
                        "disp": False,
                    },
                )
                solution_reduced = np.asarray(result.x, dtype=float)
                margin = b_ineq @ solution_reduced - rhs_ineq
                min_margin = float(np.min(margin)) if len(margin) else float("inf")
                if not np.all(np.isfinite(solution_reduced)) or min_margin < -1e-6:
                    raise RuntimeError(
                        "Forward-progress minimum-snap solve failed: "
                        f"success={result.success}, status={result.status}, "
                        f"minimum_margin={min_margin:.3e}, message={result.message}"
                    )
                self.forward_progress_solver_status = (
                    f"{constraint_mode}:{result.status}:{result.nit}"
                )

        normalized_coeffs = np.zeros((m, 3, self.N_COEFF), dtype=float)
        for axis in range(3):
            coeff_axis = particular[axis].copy()
            if free_sizes[axis] > 0:
                coeff_axis += nullspaces[axis] @ solution_reduced[
                    offsets[axis]:offsets[axis + 1]
                ]
            normalized_coeffs[:, axis, :] = coeff_axis.reshape(m, self.N_COEFF)

        constrained_values = constants.copy()
        if len(constrained_values) and total_free > 0:
            constrained_values += b_ineq @ solution_reduced
        if constraint_mode == "bernstein":
            self.forward_progress_min_bernstein_speed_m_s = (
                float(np.min(constrained_values))
                if len(constrained_values)
                else float("nan")
            )

        # Verify the actual chord-projected velocity polynomial at its exact
        # interior extrema, not merely at the sampled fallback points.  If the
        # sampled solution has a small between-sample dip, add that exact
        # extremum as a new linear QP constraint and resolve.
        def find_forward_extrema(coefficients):
            minimum = float("inf")
            violated = []
            for seg_idx, duration in enumerate(times):
                chord = waypoints[seg_idx + 1] - waypoints[seg_idx]
                chord_length = float(np.linalg.norm(chord))
                if not math.isfinite(chord_length) or chord_length < 1e-6:
                    continue
                direction = chord / chord_length
                velocity_power = np.zeros(self.POLY_ORDER, dtype=float)
                for axis in range(3):
                    velocity_power += float(direction[axis]) * (
                        np.arange(1, self.N_COEFF, dtype=float)
                        * coefficients[seg_idx, axis, 1:]
                        / float(duration)
                    )
                acceleration_power = np.arange(1, self.POLY_ORDER, dtype=float) * (
                    velocity_power[1:]
                )
                candidates = [0.0, 1.0]
                if np.any(np.abs(acceleration_power) > 1e-12):
                    for root in np.polynomial.polynomial.polyroots(acceleration_power):
                        if abs(float(np.imag(root))) <= 1e-8:
                            root_real = float(np.real(root))
                            if 0.0 < root_real < 1.0:
                                candidates.append(root_real)
                values = [
                    float(np.polynomial.polynomial.polyval(t, velocity_power))
                    for t in candidates
                ]
                local_index = int(np.argmin(values))
                local_minimum = values[local_index]
                minimum = min(minimum, local_minimum)
                if local_minimum < float(min_forward_speed_m_s) - 1e-7:
                    violated.append(
                        (seg_idx, float(candidates[local_index]), direction)
                    )
            return minimum, violated

        actual_minimum, violations = find_forward_extrema(normalized_coeffs)
        if constraint_mode == "extrema" and total_free > 0:
            refinement_count = 0
            while violations and refinement_count < 8:
                new_rows = []
                new_rhs = []
                for seg_idx, normalized_time, direction in violations:
                    duration = float(times[seg_idx])
                    basis = self._basis_row(normalized_time, 1) / duration
                    row = np.zeros(total_free, dtype=float)
                    constant = 0.0
                    segment_slice = slice(
                        seg_idx * self.N_COEFF,
                        (seg_idx + 1) * self.N_COEFF,
                    )
                    for axis in range(3):
                        weighted_basis = float(direction[axis]) * basis
                        constant += float(
                            weighted_basis @ particular[axis][segment_slice]
                        )
                        if free_sizes[axis] > 0:
                            row[offsets[axis]:offsets[axis + 1]] = (
                                weighted_basis
                                @ nullspaces[axis][segment_slice, :]
                            )
                    new_rows.append(row)
                    new_rhs.append(float(min_forward_speed_m_s) - constant)

                b_ineq = np.vstack([b_ineq, *new_rows])
                rhs_ineq = np.concatenate([rhs_ineq, np.asarray(new_rhs)])
                feasibility = linprog(
                    c=np.zeros(total_free, dtype=float),
                    A_ub=-b_ineq,
                    b_ub=-rhs_ineq,
                    bounds=[(None, None)] * total_free,
                    method="highs",
                )
                if not feasibility.success:
                    break

                def refined_objective(z: np.ndarray) -> float:
                    return float(0.5 * z @ h_regularized @ z + g_scaled @ z)

                def refined_jacobian(z: np.ndarray) -> np.ndarray:
                    return h_regularized @ z + g_scaled

                refined = minimize(
                    refined_objective,
                    np.asarray(feasibility.x, dtype=float),
                    jac=refined_jacobian,
                    method="SLSQP",
                    constraints={
                        "type": "ineq",
                        "fun": lambda z: b_ineq @ z - rhs_ineq,
                        "jac": lambda _z: b_ineq,
                    },
                    options={
                        "maxiter": int(max_iterations),
                        "ftol": 1e-10,
                        "disp": False,
                    },
                )
                solution_reduced = np.asarray(refined.x, dtype=float)
                final_margin = b_ineq @ solution_reduced - rhs_ineq
                if (
                    not np.all(np.isfinite(solution_reduced))
                    or float(np.min(final_margin)) < -1e-6
                ):
                    break
                for axis in range(3):
                    coeff_axis = particular[axis].copy()
                    if free_sizes[axis] > 0:
                        coeff_axis += nullspaces[axis] @ solution_reduced[
                            offsets[axis]:offsets[axis + 1]
                        ]
                    normalized_coeffs[:, axis, :] = coeff_axis.reshape(
                        m,
                        self.N_COEFF,
                    )
                refinement_count += 1
                actual_minimum, violations = find_forward_extrema(
                    normalized_coeffs
                )
            self.forward_progress_solver_status += f":refine={refinement_count}"

        self.forward_progress_min_speed_m_s_solved = actual_minimum
        if actual_minimum < float(min_forward_speed_m_s) - 1e-6:
            raise RuntimeError(
                "Forward-progress extrema verification failed: "
                f"minimum={actual_minimum:.6f} m/s"
            )

        # Convert p(u) coefficients to p(tau), where u = tau / T.
        coeffs = np.zeros_like(normalized_coeffs)
        powers = np.arange(self.N_COEFF, dtype=float)
        for seg_idx, duration in enumerate(times):
            coeffs[seg_idx, :, :] = normalized_coeffs[seg_idx, :, :] / (
                float(duration) ** powers
            )
        return coeffs

    def _build_normalized_global_Q(self, times: np.ndarray) -> np.ndarray:
        """Snap cost for coefficients expressed in normalized segment time."""
        m = len(times)
        q = np.zeros((self.N_COEFF * m, self.N_COEFF * m), dtype=float)
        unit_q = self._segment_Q(1.0)
        for seg_idx, duration in enumerate(times):
            segment_slice = slice(
                seg_idx * self.N_COEFF,
                (seg_idx + 1) * self.N_COEFF,
            )
            q[segment_slice, segment_slice] = unit_q / (float(duration) ** 7)
        return q

    def _build_forward_equality_constraints(
        self,
        *,
        waypoints_1d: np.ndarray,
        times: np.ndarray,
        d_start: np.ndarray,
        d_end: np.ndarray,
        waypoint_velocities: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the underdetermined smoothness constraints for the joint QP.

        Start velocity/acceleration/jerk and terminal velocity are retained.
        Terminal acceleration/jerk and continuity above jerk are intentionally
        left to the snap objective so the inequality-constrained solve keeps
        enough freedom for a fast multi-gate route.
        """
        m = len(times)
        n = self.N_COEFF * m
        rows = []
        values = []

        def add(segment_idx: int, basis: np.ndarray, value: float) -> None:
            row = np.zeros(n, dtype=float)
            segment_slice = slice(
                segment_idx * self.N_COEFF,
                (segment_idx + 1) * self.N_COEFF,
            )
            row[segment_slice] = basis
            rows.append(row)
            values.append(float(value))

        for seg_idx, duration in enumerate(times):
            add(seg_idx, self._basis_row(0.0, 0), waypoints_1d[seg_idx])
            add(seg_idx, self._basis_row(1.0, 0), waypoints_1d[seg_idx + 1])

        for derivative, target in zip((1, 2, 3), d_start):
            add(
                0,
                self._basis_row(0.0, derivative) / (float(times[0]) ** derivative),
                target,
            )
        add(
            m - 1,
            self._basis_row(1.0, 1) / float(times[-1]),
            d_end[0],
        )

        constrained_internal_velocity = set()
        if waypoint_velocities is not None:
            velocities = np.asarray(waypoint_velocities, dtype=float).reshape(-1)
            if len(velocities) != len(waypoints_1d):
                raise ValueError(
                    "`waypoint_velocities` length must match waypoints length "
                    f"{len(waypoints_1d)}, got {len(velocities)}"
                )
            constrained_internal_velocity = {
                idx for idx in range(1, len(velocities) - 1)
                if np.isfinite(velocities[idx])
            }

        for seg_idx in range(m - 1):
            next_idx = seg_idx + 1
            left_duration = float(times[seg_idx])
            right_duration = float(times[next_idx])
            left_slice = slice(
                seg_idx * self.N_COEFF,
                (seg_idx + 1) * self.N_COEFF,
            )
            right_slice = slice(
                next_idx * self.N_COEFF,
                (next_idx + 1) * self.N_COEFF,
            )

            if next_idx in constrained_internal_velocity:
                target = float(waypoint_velocities[next_idx])
                add(
                    seg_idx,
                    self._basis_row(1.0, 1) / left_duration,
                    target,
                )
                add(
                    next_idx,
                    self._basis_row(0.0, 1) / right_duration,
                    target,
                )
            else:
                row = np.zeros(n, dtype=float)
                row[left_slice] = self._basis_row(1.0, 1) / left_duration
                row[right_slice] = -self._basis_row(0.0, 1) / right_duration
                rows.append(row)
                values.append(0.0)

            for derivative in (2, 3):
                row = np.zeros(n, dtype=float)
                row[left_slice] = self._basis_row(1.0, derivative) / (
                    left_duration ** derivative
                )
                row[right_slice] = -self._basis_row(0.0, derivative) / (
                    right_duration ** derivative
                )
                rows.append(row)
                values.append(0.0)

        return np.vstack(rows), np.asarray(values, dtype=float)

    @classmethod
    def _velocity_bernstein_row(cls, index: int, duration: float) -> np.ndarray:
        """Map normalized power coefficients to one degree-6 velocity control point."""
        degree = cls.POLY_ORDER - 1
        if not 0 <= int(index) <= degree:
            raise ValueError(f"Bernstein index must be in [0, {degree}], got {index}")
        row = np.zeros(cls.N_COEFF, dtype=float)
        for power in range(int(index) + 1):
            conversion = math.comb(int(index), power) / math.comb(degree, power)
            row[power + 1] = (power + 1) * conversion / float(duration)
        return row

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

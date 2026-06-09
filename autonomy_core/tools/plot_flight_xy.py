#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from autonomy_core.planning.minimum_snap_planner_multi_time_optimized import (
        MultiSegmentMinimumSnapPlanner,
    )
except ModuleNotFoundError as exc:
    if exc.name != "autonomy_core":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autonomy_core.planning.minimum_snap_planner_multi_time_optimized import (
        MultiSegmentMinimumSnapPlanner,
    )


GATES = {
    1: (0.0, 8.0, 1.5),
    2: (0.8, 16.0, 1.5),
    3: (-0.8, 24.0, 1.5),
}


def num(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def nonempty(series):
    return series.fillna("").astype(str).str.strip() != ""


def speed_xy(df, xcol, ycol, tcol="t"):
    t = num(df, tcol).to_numpy()
    x = num(df, xcol).to_numpy()
    y = num(df, ycol).to_numpy()

    good = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    v = np.full(len(df), np.nan)

    if good.sum() < 3:
        return pd.Series(v, index=df.index)

    tg = t[good]
    xg = x[good]
    yg = y[good]

    dxdt = np.gradient(xg, tg)
    dydt = np.gradient(yg, tg)
    vg = np.sqrt(dxdt**2 + dydt**2)

    v[np.where(good)[0]] = vg
    return pd.Series(v, index=df.index)


def get_replan_mask(df):
    # Prefer true replan_requested events if available.
    if "replan_requested" in df.columns:
        s = df["replan_requested"].fillna("").astype(str).str.lower()
        return s.isin(["true", "1", "yes"])

    # Fallback: non-empty reason.
    replan_reason = df["replan_reason"] if "replan_reason" in df.columns else pd.Series("", index=df.index)
    return nonempty(replan_reason)


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_planning_waypoints(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    points = []
    for item in str(value).strip().split(";"):
        if not item:
            continue
        _, separator, coordinates = item.partition(":")
        if not separator:
            return None
        try:
            point = [float(component) for component in coordinates.split(",")]
        except ValueError:
            return None
        if len(point) != 3 or not np.all(np.isfinite(point)):
            return None
        points.append(point)

    if len(points) < 2:
        return None
    return np.asarray(points, dtype=float)


def choose_segment_time(p0, v0, p1, vmax=2.5, amax=2.0, t_min=1.0):
    delta = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        return t_min

    direction = delta / distance
    v_along = float(np.dot(np.asarray(v0, dtype=float), direction))
    t_acc = vmax / amax
    d_acc = 0.5 * amax * t_acc**2

    if distance > 2.0 * d_acc:
        t_base = 2.0 * t_acc + (distance - 2.0 * d_acc) / vmax
    else:
        t_base = 2.0 * np.sqrt(distance / amax)

    if v_along < 0.0:
        t_base += min(abs(v_along) / amax, 2.0)
    else:
        t_base -= min(v_along / (2.0 * amax), 0.5)
    return max(float(t_base), t_min)


def reconstruct_future_spline(df, current_idx, samples=240):
    if "planning_horizon_waypoints" not in df.columns:
        return None

    horizon = str(df.loc[current_idx, "planning_horizon_waypoints"]).strip()
    waypoints = parse_planning_waypoints(horizon)
    if waypoints is None:
        return None

    current_pos = df.index.get_loc(current_idx)
    plan_start_pos = current_pos
    while plan_start_pos > 0:
        previous = str(
            df.iloc[plan_start_pos - 1].get("planning_horizon_waypoints", "")
        ).strip()
        if previous != horizon:
            break
        plan_start_pos -= 1

    plan_row = df.iloc[plan_start_pos]
    current_row = df.loc[current_idx]
    start_velocity = np.nan_to_num(
        np.array(
            [
                plan_row.get("vx", 0.0),
                plan_row.get("vy", 0.0),
                plan_row.get("vz", 0.0),
            ],
            dtype=float,
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    segment_times = np.asarray(
        [
            choose_segment_time(
                waypoints[i],
                start_velocity if i == 0 else np.zeros(3, dtype=float),
                waypoints[i + 1],
            )
            for i in range(len(waypoints) - 1)
        ],
        dtype=float,
    )

    waypoint_velocities = np.full_like(waypoints, np.nan, dtype=float)
    passthrough_enabled = parse_bool(
        current_row.get("passthrough_velocity_enabled", False)
    )
    speed = pd.to_numeric(
        pd.Series([current_row.get("passthrough_speed_used", np.nan)]),
        errors="coerce",
    ).iloc[0]
    if passthrough_enabled and np.isfinite(speed) and speed > 0.0:
        for i in range(1, len(waypoints) - 1):
            direction = waypoints[i + 1] - waypoints[i - 1]
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                waypoint_velocities[i] = float(speed) * direction / norm

    planner = MultiSegmentMinimumSnapPlanner()
    try:
        planner.update(
            waypoints=waypoints,
            times=segment_times,
            v_start=start_velocity,
            v_end=np.zeros(3, dtype=float),
            a_start=np.zeros(3, dtype=float),
            a_end=np.zeros(3, dtype=float),
            j_start=np.zeros(3, dtype=float),
            j_end=np.zeros(3, dtype=float),
            waypoint_velocities=waypoint_velocities,
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None

    plan_t = float(pd.to_numeric(pd.Series([plan_row.get("t")]), errors="coerce").iloc[0])
    current_t = float(
        pd.to_numeric(pd.Series([current_row.get("t")]), errors="coerce").iloc[0]
    )
    if not (np.isfinite(plan_t) and np.isfinite(current_t)):
        return None

    elapsed = max(0.0, current_t - plan_t)
    if elapsed >= planner.total_time:
        return None

    sample_times = np.linspace(elapsed, planner.total_time, max(2, int(samples)))
    future = np.asarray([planner.sample(t)[0] for t in sample_times], dtype=float)
    return future, waypoints


def all_planned_waypoints(df):
    if "planning_horizon_waypoints" not in df.columns:
        return np.empty((0, 3), dtype=float)

    point_sets = []
    for value in df["planning_horizon_waypoints"].dropna().unique():
        points = parse_planning_waypoints(value)
        if points is not None:
            point_sets.append(points)
    if not point_sets:
        return np.empty((0, 3), dtype=float)
    return np.vstack(point_sets)


def load_installed_plan_samples(csv_path):
    plans_path = csv_path.with_name(f"{csv_path.stem}_plans.csv")
    if not plans_path.exists():
        return None

    plan_df = pd.read_csv(plans_path)
    required = {"plan_id", "x", "y"}
    if not required.issubset(plan_df.columns):
        return None
    plan_df["plan_id"] = pd.to_numeric(plan_df["plan_id"], errors="coerce")
    plan_df["x"] = pd.to_numeric(plan_df["x"], errors="coerce")
    plan_df["y"] = pd.to_numeric(plan_df["y"], errors="coerce")
    return plan_df


def installed_plan_for_row(df, plan_df, current_idx):
    if plan_df is None or "active_plan_id" not in df.columns:
        return None
    plan_id = pd.to_numeric(
        pd.Series([df.loc[current_idx, "active_plan_id"]]),
        errors="coerce",
    ).iloc[0]
    if not np.isfinite(plan_id) or int(plan_id) <= 0:
        return None

    rows = plan_df[plan_df["plan_id"] == int(plan_id)]
    rows = rows[np.isfinite(rows["x"]) & np.isfinite(rows["y"])]
    if rows.empty:
        return None
    return int(plan_id), rows


def draw_frame(
    df,
    plan_df,
    csv_path,
    out_path,
    t_now,
    annotate=False,
    xlim=None,
    ylim=None,
    tracking_only=False,
    show_reconstructed_spline=False,
):
    t = num(df, "t")

    px = num(df, "px")
    py = num(df, "py")

    p_ref_x = num(df, "p_ref_x")
    p_ref_y = num(df, "p_ref_y")

    target_x = num(df, "target_x")
    target_y = num(df, "target_y")

    v_ref_x = num(df, "v_ref_x")
    v_ref_y = num(df, "v_ref_y")
    v_ref_speed = np.sqrt(v_ref_x**2 + v_ref_y**2)
    tracking_err_norm = num(df, "tracking_err_norm")

    actual_speed = speed_xy(df, "px", "py")

    replan_mask = get_replan_mask(df)

    target_delta = num(df, "target_update_delta_m")
    target_shift_mask = target_delta.fillna(0) > 0.02

    low_vref_mask = v_ref_speed < 0.5
    low_actual_mask = actual_speed < 0.5

    past = t <= t_now
    current_idx = t[past].last_valid_index()
    if current_idx is None:
        return

    fig, ax = plt.subplots(figsize=(9, 12))

    if not tracking_only:
        # Full ghost path for context
        ax.plot(px, py, linewidth=1, alpha=0.18, label="full vehicle path")

    # Vehicle path up to now
    ax.plot(px[past], py[past], linewidth=2.5, label="vehicle path so far")

    # Reference path up to now
    ref_good = past & np.isfinite(p_ref_x) & np.isfinite(p_ref_y)
    ax.plot(p_ref_x[ref_good], p_ref_y[ref_good], linewidth=1.5, alpha=0.85, label="reference p_ref so far")

    if not tracking_only:
        installed_plan = installed_plan_for_row(df, plan_df, current_idx)
        if installed_plan is not None:
            plan_id, plan_rows = installed_plan
            ax.plot(
                plan_rows["x"],
                plan_rows["y"],
                linewidth=2.5,
                linestyle="--",
                color="tab:purple",
                label=f"installed planner trajectory plan_id={plan_id}",
            )
        elif show_reconstructed_spline:
            future_plan = reconstruct_future_spline(df, current_idx)
            if future_plan is not None:
                future, planned_waypoints = future_plan
                ax.plot(
                    future[:, 0],
                    future[:, 1],
                    linewidth=2.0,
                    linestyle="--",
                    color="tab:purple",
                    alpha=0.55,
                    label="reconstructed horizon approximation",
                )
                ax.scatter(
                    planned_waypoints[1:, 0],
                    planned_waypoints[1:, 1],
                    s=70,
                    marker="x",
                    color="tab:purple",
                    alpha=0.55,
                    label="reconstructed waypoint approximation",
                )

        # Active target samples up to now
        tgt_good = past & np.isfinite(target_x) & np.isfinite(target_y)
        ax.scatter(
            target_x[tgt_good],
            target_y[tgt_good],
            s=9,
            alpha=0.35,
            label="active target samples",
        )

    # Current vehicle / reference / target markers
    cx = px.loc[current_idx]
    cy = py.loc[current_idx]
    crx = p_ref_x.loc[current_idx]
    cry = p_ref_y.loc[current_idx]
    ctx = target_x.loc[current_idx]
    cty = target_y.loc[current_idx]

    if np.isfinite(cx) and np.isfinite(cy):
        ax.scatter([cx], [cy], s=130, marker="*", label="current vehicle")

    if np.isfinite(crx) and np.isfinite(cry):
        ax.scatter([crx], [cry], s=100, marker="D", label="current p_ref")

    if not tracking_only and np.isfinite(ctx) and np.isfinite(cty):
        ax.scatter([ctx], [cty], s=120, marker="P", label="current target")

    # Draw line from vehicle to current reference
    if all(np.isfinite(v) for v in [cx, cy, crx, cry]):
        ax.plot([cx, crx], [cy, cry], linewidth=1.5, linestyle="--", label="vehicle → p_ref")

    if not tracking_only:
        # Gate centers
        for gate_id, (gx, gy, gz) in GATES.items():
            ax.scatter([gx], [gy], marker="s", s=140, label="gate centers" if gate_id == 1 else None)
            ax.text(gx + 0.12, gy + 0.12, f"G{gate_id}", fontsize=11)

        # Replan events so far
        rp = df[past & replan_mask]
        ax.scatter(num(rp, "px"), num(rp, "py"), marker="x", s=80, label="real replan event")

        # Target update/shift events so far
        sh = df[past & target_shift_mask]
        ax.scatter(num(sh, "px"), num(sh, "py"), marker="o", facecolors="none", s=90, label="target update/shift")

        # Low reference speed zones so far
        lv = df[past & low_vref_mask.fillna(False)]
        ax.scatter(num(lv, "px"), num(lv, "py"), marker=".", s=35, label="v_ref < 0.5 m/s")

        # Low actual speed zones so far
        la = df[past & low_actual_mask.fillna(False)]
        ax.scatter(num(la, "px"), num(la, "py"), marker=".", s=12, alpha=0.6, label="actual xy speed < 0.5 m/s")

    if annotate:
        last_xy = None
        for idx, row in rp.iterrows():
            x = float(row.get("px", np.nan))
            y = float(row.get("py", np.nan))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue

            reason = str(row.get("replan_reason", "")).strip()
            tt = row.get("t", np.nan)

            if last_xy is not None and math.hypot(x - last_xy[0], y - last_xy[1]) < 0.45:
                continue
            last_xy = (x, y)

            label = f"{float(tt):.2f}s\n{reason}"
            ax.text(x + 0.08, y + 0.08, label, fontsize=7)

    # Current text panel
    active_gate = df.loc[current_idx, "active_gate_idx"] if "active_gate_idx" in df.columns else ""
    active_track = df.loc[current_idx, "active_target_track_id"] if "active_target_track_id" in df.columns else ""
    replan_reason = df.loc[current_idx, "replan_reason"] if "replan_reason" in df.columns else ""
    plan_retiming_applied = (
        df.loc[current_idx, "plan_retiming_applied"]
        if "plan_retiming_applied" in df.columns
        else ""
    )
    tracker_acc_xy_limited = (
        df.loc[current_idx, "tracker_acc_xy_limited"]
        if "tracker_acc_xy_limited" in df.columns
        else ""
    )
    vref_now = float(v_ref_speed.loc[current_idx]) if np.isfinite(v_ref_speed.loc[current_idx]) else np.nan
    actual_now = float(actual_speed.loc[current_idx]) if np.isfinite(actual_speed.loc[current_idx]) else np.nan
    xy_err = (
        math.hypot(float(crx) - float(cx), float(cry) - float(cy))
        if all(np.isfinite(v) for v in [cx, cy, crx, cry])
        else float("nan")
    )
    tracking_err_3d = float(tracking_err_norm.loc[current_idx])

    panel = (
        f"t = {float(t.loc[current_idx]):.2f}s\n"
        f"active_gate_idx = {active_gate}\n"
        f"active_target_track_id = {active_track}\n"
        f"xy_err = {xy_err:.3f} m\n"
        f"tracking_err_3d = {tracking_err_3d:.3f} m\n"
        f"v_ref_xy = {vref_now:.2f} m/s\n"
        f"actual_xy = {actual_now:.2f} m/s\n"
        f"plan_retiming_applied = {plan_retiming_applied}\n"
        f"tracker_acc_xy_limited = {tracker_acc_xy_limited}\n"
        f"replan_reason = {replan_reason}"
    )

    ax.text(
        0.02, 0.98, panel,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.25),
    )

    title_mode = "Tracking-only XY" if tracking_only else "Flight XY"
    ax.set_title(f"{title_mode} up to t={t_now:.1f}s: {csv_path.name}")
    ax.set_xlabel("world x / lateral [m]")
    ax.set_ylabel("world y / course-forward [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="flight_log.csv path")
    ap.add_argument("--out-dir", default=None, help="output directory for PNG frames")
    ap.add_argument("--step", type=float, default=1.0, help="seconds between output frames")
    ap.add_argument("--annotate", action="store_true", help="annotate real replan events")
    ap.add_argument("--start", type=float, default=None, help="start time override")
    ap.add_argument("--end", type=float, default=None, help="end time override")
    ap.add_argument(
        "--tracking-only",
        action="store_true",
        help="show only historical vehicle and p_ref tracking, hiding future/context layers",
    )
    ap.add_argument(
        "--show-reconstructed-spline",
        action="store_true",
        help="show the old offline reconstructed horizon approximation when no sidecar plan is available",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    plan_df = load_installed_plan_samples(csv_path)

    t = num(df, "t")
    finite_t = t[np.isfinite(t)]

    if finite_t.empty:
        raise RuntimeError("No valid t column found.")

    t0 = float(finite_t.min()) if args.start is None else args.start
    t1 = float(finite_t.max()) if args.end is None else args.end

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.with_suffix("").parent / f"{csv_path.stem}_xy_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed plot limits so frames don't jump around.
    px = num(df, "px")
    py = num(df, "py")
    p_ref_x = num(df, "p_ref_x")
    p_ref_y = num(df, "p_ref_y")
    target_x = num(df, "target_x")
    target_y = num(df, "target_y")
    planned_waypoints = all_planned_waypoints(df)
    plan_x = (
        plan_df["x"]
        if plan_df is not None and "x" in plan_df.columns
        else pd.Series(dtype=float)
    )
    plan_y = (
        plan_df["y"]
        if plan_df is not None and "y" in plan_df.columns
        else pd.Series(dtype=float)
    )

    all_x = pd.concat([
        px,
        p_ref_x,
        target_x,
        plan_x,
        pd.Series([g[0] for g in GATES.values()]),
        pd.Series(planned_waypoints[:, 0]),
    ])
    all_y = pd.concat([
        py,
        p_ref_y,
        target_y,
        plan_y,
        pd.Series([g[1] for g in GATES.values()]),
        pd.Series(planned_waypoints[:, 1]),
    ])

    x_min, x_max = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))

    x_pad = max(0.5, 0.1 * (x_max - x_min))
    y_pad = max(0.5, 0.1 * (y_max - y_min))

    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    times = np.arange(t0, t1 + 1e-9, args.step)

    for i, tn in enumerate(times):
        out_path = out_dir / f"xy_{i:04d}_t{tn:06.2f}.png"
        draw_frame(
            df=df,
            plan_df=plan_df,
            csv_path=csv_path,
            out_path=out_path,
            t_now=float(tn),
            annotate=args.annotate,
            xlim=xlim,
            ylim=ylim,
            tracking_only=args.tracking_only,
            show_reconstructed_spline=args.show_reconstructed_spline,
        )
        print(f"saved {out_path}")

    print(f"\nSaved {len(times)} frames to: {out_dir}")


if __name__ == "__main__":
    main()

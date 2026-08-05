#!/usr/bin/env python3
"""Test whether VQ1 HIGHRES_IMU yacc has a recoverable scale/sign error.

Ground-truth LOCAL_POSITION_NED and the verified physical ATTITUDE boundary
are used only to derive an offline body-frame specific-force target. Candidate
corrections consume the recorded yacc value alone. Each training-bias and
fitted-affine result is evaluated leave-one-run-out. World-y acceleration from
rolling the body does not by itself excite body-frame yacc.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from diagnose_openvins_replay import (
    DEFAULT_SMOOTHING_WINDOW_S,
    GRAVITY_NED_M_S2,
    DiagnosticError,
    _centered_moving_average,
    _differentiate,
    _read_numeric_csv,
    _require_increasing,
    _vector_rmse,
    _window_samples,
    euler_body_frd_to_ned_matrix,
    nearest_rotation_samples,
)


@dataclass(frozen=True)
class RunSignals:
    name: str
    dataset: str
    measured_y: np.ndarray
    expected_y: np.ndarray
    velocity_frame: str
    raw_velocity_rmse_m_s: float
    body_velocity_rmse_m_s: float | None


def _deduplicate_last(rows: np.ndarray) -> np.ndarray:
    """Sort by the first column and keep the last row for each timestamp."""
    by_timestamp = {int(row[0]): row for row in rows}
    return np.asarray([by_timestamp[key] for key in sorted(by_timestamp)], dtype=float)


def _load_run(dataset: Path, smoothing_window_s: float) -> RunSignals:
    imu = _deduplicate_last(
        _read_numeric_csv(
            dataset / "imu" / "data.csv",
            ("time_usec", "xacc", "yacc", "zacc"),
        )
    )
    truth = _deduplicate_last(
        _read_numeric_csv(
            dataset / "truth" / "local_position_ned.csv",
            ("time_boot_ms", "x", "y", "z", "vx", "vy", "vz"),
        )
    )
    attitude = _deduplicate_last(
        _read_numeric_csv(
            dataset / "truth" / "attitude.csv",
            ("time_boot_ms", "roll", "pitch", "yaw"),
        )
    )

    imu_time = imu[:, 0] * 1e-6
    truth_time = truth[:, 0] * 1e-3
    measured = imu[:, 1:4]
    truth_position = truth[:, 1:4]
    reported_velocity = truth[:, 4:7]
    attitude_time = attitude[:, 0] * 1e-3
    attitude_rotations = euler_body_frd_to_ned_matrix(
        attitude[:, 1],
        -attitude[:, 2],
        attitude[:, 3],
    )
    body_to_ned, _attitude_delta_s = nearest_rotation_samples(
        attitude_time,
        attitude_rotations,
        truth_time,
    )
    _require_increasing(imu_time, "IMU")
    _require_increasing(truth_time, "truth")

    velocity_check_samples = _window_samples(truth_time, 0.50)
    position_rate = _centered_moving_average(
        _differentiate(truth_position, truth_time), velocity_check_samples
    )
    velocity_check_interior = (
        (truth_time >= truth_time[0] + 0.50)
        & (truth_time <= truth_time[-1] - 0.50)
    )
    raw_velocity_rmse = _vector_rmse(
        reported_velocity[velocity_check_interior],
        position_rate[velocity_check_interior],
    )
    body_velocity_rmse = None
    velocity_frame = "local_ned"
    velocity_ned = reported_velocity

    truth_smoothing_samples = _window_samples(truth_time, smoothing_window_s)
    velocity_ned_smoothed = _centered_moving_average(
        velocity_ned, truth_smoothing_samples
    )
    acceleration_ned = _differentiate(velocity_ned_smoothed, truth_time)
    specific_force_body = np.einsum(
        "nji,nj->ni", body_to_ned, acceleration_ned - GRAVITY_NED_M_S2
    )

    overlap = (imu_time >= truth_time[0]) & (imu_time <= truth_time[-1])
    if int(np.count_nonzero(overlap)) < 100:
        raise DiagnosticError(f"{dataset}: fewer than 100 overlapping IMU/truth samples")
    overlap_time = imu_time[overlap]
    overlap_measured = measured[overlap]
    overlap_expected = np.column_stack(
        [
            np.interp(overlap_time, truth_time, specific_force_body[:, axis])
            for axis in range(3)
        ]
    )
    imu_smoothing_samples = _window_samples(overlap_time, smoothing_window_s)
    overlap_measured = _centered_moving_average(
        overlap_measured, imu_smoothing_samples
    )
    overlap_expected = _centered_moving_average(
        overlap_expected, imu_smoothing_samples
    )
    interior = (
        (overlap_time >= overlap_time[0] + smoothing_window_s)
        & (overlap_time <= overlap_time[-1] - smoothing_window_s)
    )
    if int(np.count_nonzero(interior)) < 100:
        raise DiagnosticError(f"{dataset}: insufficient samples after edge trimming")

    return RunSignals(
        name=dataset.parent.name,
        dataset=str(dataset),
        measured_y=overlap_measured[interior, 1],
        expected_y=overlap_expected[interior, 1],
        velocity_frame=velocity_frame,
        raw_velocity_rmse_m_s=raw_velocity_rmse,
        body_velocity_rmse_m_s=body_velocity_rmse,
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    rmse = float(np.sqrt(np.mean((predicted - expected) ** 2)))
    expected_std = float(np.std(expected))
    return {
        "samples": int(expected.size),
        "rmse_m_s2": rmse,
        "normalized_rmse": rmse / expected_std if expected_std > 1e-12 else None,
        "correlation": _correlation(predicted, expected),
        "predicted_mean_m_s2": float(np.mean(predicted)),
        "predicted_std_m_s2": float(np.std(predicted)),
        "expected_mean_m_s2": float(np.mean(expected)),
        "expected_std_m_s2": expected_std,
    }


def _aggregate_fold_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_run_normalized_rmse": float(
            np.mean([fold["metrics"]["normalized_rmse"] for fold in folds])
        ),
        "worst_run_normalized_rmse": float(
            np.max([fold["metrics"]["normalized_rmse"] for fold in folds])
        ),
        "mean_run_rmse_m_s2": float(
            np.mean([fold["metrics"]["rmse_m_s2"] for fold in folds])
        ),
    }


def _score_fixed_gain(runs: list[RunSignals], gain: float) -> dict[str, Any]:
    zero_bias_folds = [
        {
            "run": run.name,
            "bias_m_s2": 0.0,
            "metrics": _metrics(gain * run.measured_y, run.expected_y),
        }
        for run in runs
    ]

    held_out_folds: list[dict[str, Any]] = []
    for held_out in runs:
        training_runs = [run for run in runs if run is not held_out]
        training_measured = np.concatenate([run.measured_y for run in training_runs])
        training_expected = np.concatenate([run.expected_y for run in training_runs])
        bias = float(np.mean(training_expected - gain * training_measured))
        held_out_folds.append(
            {
                "run": held_out.name,
                "bias_m_s2": bias,
                "metrics": _metrics(
                    gain * held_out.measured_y + bias,
                    held_out.expected_y,
                ),
            }
        )

    return {
        "gain": gain,
        "zero_bias": {
            **_aggregate_fold_metrics(zero_bias_folds),
            "runs": zero_bias_folds,
        },
        "leave_one_run_out_training_bias": {
            **_aggregate_fold_metrics(held_out_folds),
            "runs": held_out_folds,
        },
    }


def _fit_affine(measured: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((measured, np.ones(measured.size)))
    gain, bias = np.linalg.lstsq(design, expected, rcond=None)[0]
    return float(gain), float(bias)


def _score_affine(runs: list[RunSignals]) -> dict[str, Any]:
    all_measured = np.concatenate([run.measured_y for run in runs])
    all_expected = np.concatenate([run.expected_y for run in runs])
    global_gain, global_bias = _fit_affine(all_measured, all_expected)

    per_run = []
    held_out_folds = []
    for held_out in runs:
        run_gain, run_bias = _fit_affine(held_out.measured_y, held_out.expected_y)
        per_run.append(
            {
                "run": held_out.name,
                "gain": run_gain,
                "bias_m_s2": run_bias,
                "metrics": _metrics(
                    run_gain * held_out.measured_y + run_bias,
                    held_out.expected_y,
                ),
            }
        )

        training_runs = [run for run in runs if run is not held_out]
        training_measured = np.concatenate([run.measured_y for run in training_runs])
        training_expected = np.concatenate([run.expected_y for run in training_runs])
        gain, bias = _fit_affine(training_measured, training_expected)
        held_out_folds.append(
            {
                "run": held_out.name,
                "training_gain": gain,
                "training_bias_m_s2": bias,
                "metrics": _metrics(
                    gain * held_out.measured_y + bias,
                    held_out.expected_y,
                ),
            }
        )

    return {
        "global_in_sample": {
            "gain": global_gain,
            "bias_m_s2": global_bias,
            "metrics": _metrics(
                global_gain * all_measured + global_bias,
                all_expected,
            ),
        },
        "per_run_in_sample": per_run,
        "leave_one_run_out": {
            **_aggregate_fold_metrics(held_out_folds),
            "runs": held_out_folds,
        },
    }


def run_sweep(
    datasets: Iterable[Path],
    gains: Iterable[float],
    smoothing_window_s: float,
) -> dict[str, Any]:
    runs: list[RunSignals] = []
    excluded_runs: list[dict[str, str]] = []
    for path in datasets:
        resolved = path.resolve()
        try:
            runs.append(_load_run(resolved, smoothing_window_s))
        except DiagnosticError as exc:
            excluded_runs.append(
                {
                    "dataset": str(resolved),
                    "reason": str(exc),
                }
            )
    if len(runs) < 2:
        raise DiagnosticError("at least two runs are required for held-out validation")

    run_reports = []
    for run in runs:
        run_reports.append(
            {
                "run": run.name,
                "dataset": run.dataset,
                "samples": int(run.expected_y.size),
                "velocity_frame": run.velocity_frame,
                "raw_velocity_rmse_m_s": run.raw_velocity_rmse_m_s,
                "body_velocity_rmse_m_s": run.body_velocity_rmse_m_s,
                "measured_y_mean_m_s2": float(np.mean(run.measured_y)),
                "measured_y_std_m_s2": float(np.std(run.measured_y)),
                "expected_y_mean_m_s2": float(np.mean(run.expected_y)),
                "expected_y_std_m_s2": float(np.std(run.expected_y)),
                "measured_expected_correlation": _correlation(
                    run.measured_y, run.expected_y
                ),
            }
        )

    candidates = [_score_fixed_gain(runs, float(gain)) for gain in gains]
    candidates.sort(
        key=lambda item: item["leave_one_run_out_training_bias"][
            "mean_run_normalized_rmse"
        ]
    )
    return {
        "format": "aigp_yacc_scale_sweep",
        "format_version": 1,
        "truth_usage": "offline scoring only; never supplied as a runtime input",
        "candidate_model": "corrected_yacc = gain * recorded_yacc + training_bias",
        "smoothing_window_s": smoothing_window_s,
        "excluded_runs": excluded_runs,
        "runs": run_reports,
        "fixed_gain_candidates_ranked": candidates,
        "affine_fit": _score_affine(runs),
    }


def _default_datasets(runs_root: Path) -> list[Path]:
    return sorted(
        path / "vio_dataset"
        for path in runs_root.glob("vq1_openvins_*")
        if (path / "vio_dataset" / "imu" / "data.csv").is_file()
        and (path / "vio_dataset" / "truth" / "local_position_ned.csv").is_file()
        and (path / "vio_dataset" / "truth" / "attitude.csv").is_file()
    )


def _parse_gains(text: str) -> list[float]:
    if text.strip().lower() == "integer:-200:200":
        return [float(value) for value in range(-200, 201)]
    try:
        gains = [float(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid gain list: {exc}") from exc
    if not gains or any(not math.isfinite(gain) for gain in gains):
        raise argparse.ArgumentTypeError("gain list must contain finite values")
    return gains


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs" / "runs",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        help="explicit vio_dataset path; repeat to select multiple runs",
    )
    parser.add_argument(
        "--gains",
        default="integer:-200:200",
        help="comma-separated gains, or integer:-200:200 (default)",
    )
    parser.add_argument(
        "--smoothing-window-s",
        type=float,
        default=DEFAULT_SMOOTHING_WINDOW_S,
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoothing_window_s <= 0.0:
        raise DiagnosticError("--smoothing-window-s must be positive")
    datasets = args.dataset or _default_datasets(args.runs_root.resolve())
    report = run_sweep(
        datasets,
        _parse_gains(args.gains),
        args.smoothing_window_s,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    print("Runs:")
    for run in report["runs"]:
        print(
            f"  {run['run']}: expected std={run['expected_y_std_m_s2']:.4f}, "
            f"measured std={run['measured_y_std_m_s2']:.4f}, "
            f"corr={run['measured_expected_correlation']:.4f}"
        )
    for excluded in report["excluded_runs"]:
        print(f"  excluded {excluded['dataset']}: {excluded['reason']}")
    print("Best fixed gains by leave-one-run-out normalized RMSE:")
    for candidate in report["fixed_gain_candidates_ranked"][:10]:
        score = candidate["leave_one_run_out_training_bias"]
        print(
            f"  gain={candidate['gain']:+.0f}: mean={score['mean_run_normalized_rmse']:.4f}, "
            f"worst={score['worst_run_normalized_rmse']:.4f}"
        )
    for requested_gain in (100.0, -100.0, 0.0):
        candidate = next(
            (
                item
                for item in report["fixed_gain_candidates_ranked"]
                if item["gain"] == requested_gain
            ),
            None,
        )
        if candidate:
            score = candidate["leave_one_run_out_training_bias"]
            print(
                f"gain={requested_gain:+.0f}: mean held-out normalized RMSE="
                f"{score['mean_run_normalized_rmse']:.4f}"
            )
    affine = report["affine_fit"]
    print(
        "Global in-sample affine: "
        f"gain={affine['global_in_sample']['gain']:.4f}, "
        f"bias={affine['global_in_sample']['bias_m_s2']:.4f} m/s^2"
    )
    print(
        "Affine leave-one-run-out: mean normalized RMSE="
        f"{affine['leave_one_run_out']['mean_run_normalized_rmse']:.4f}, "
        f"worst={affine['leave_one_run_out']['worst_run_normalized_rmse']:.4f}"
    )
    if args.output:
        print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DiagnosticError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

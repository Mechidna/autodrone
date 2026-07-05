import numpy as np

from target_manager import TargetManager


def test_active_target_lock_holds_current_gate_while_live_center_moves():
    manager = TargetManager(race_gate_count=3, active_target_lost_grace_s=2.0)
    locked = manager.lock_target(
        gate_idx=0,
        track_id=10,
        center_neu=np.array([0.0, 8.0, 1.2]),
        reason="unit_test",
        now_s=1.0,
    )

    gates, track_ids = manager.update_live_targets(
        gate_idx=0,
        gates=[
            np.array([0.8, 8.7, 1.0]),
            np.array([1.0, 16.0, 1.2]),
        ],
        track_ids=[10, 11],
        now_s=1.2,
    )

    assert track_ids == [10, 11]
    np.testing.assert_allclose(gates[0], locked)
    np.testing.assert_allclose(gates[1], np.array([1.0, 16.0, 1.2]))

    diag = manager.diagnostics(now_s=1.2)
    assert diag.locked
    assert diag.suppress_active_replan
    assert diag.shift_m > 1.0
    np.testing.assert_allclose(diag.center_at_plan, np.array([0.0, 8.0, 1.2]))
    np.testing.assert_allclose(diag.latest_center, np.array([0.8, 8.7, 1.0]))


def test_mark_passed_advances_gate_and_clears_active_lock():
    manager = TargetManager(race_gate_count=3, active_target_lost_grace_s=2.0)
    manager.lock_target(
        gate_idx=1,
        track_id=22,
        center_neu=np.array([0.5, 16.0, 1.1]),
        reason="unit_test",
        now_s=1.0,
    )

    manager.mark_passed(pos_neu=np.array([0.5, 16.0, 1.1]), distance_m=0.0)

    diag = manager.diagnostics(now_s=2.0)
    assert manager.current_gate_idx == 2
    assert 22 in manager.completed_track_ids
    assert not diag.locked
    assert diag.active_track_id is None

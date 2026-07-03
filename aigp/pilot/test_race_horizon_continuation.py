import numpy as np

from autonomy_wrapper import PyAIPilotAutonomyAPI


def test_gate_pass_advances_inside_existing_horizon():
    api = PyAIPilotAutonomyAPI(use_perception=False, race_gate_count=4)
    api.horizon_continuation_enabled = True
    api.terminal_velocity_enabled = True
    api.terminal_speed_m_s = 0.5
    api.gate_centers_neu = [
        np.array([0.0, 10.0, 1.5]),
        np.array([0.0, 20.0, 1.5]),
        np.array([0.0, 30.0, 1.5]),
    ]
    api.gate_track_ids = [-1, -2, -3]
    api.current_gate_idx = 0

    planned = api._path_plan(
        pos=np.array([0.0, 0.0, 1.5]),
        vel=np.zeros(3),
    )

    assert planned
    assert api.active_plan_mode == "gate_horizon"
    assert api.active_horizon_gate_indices == [0, 1, 2]
    assert np.linalg.norm(api.active_terminal_velocity) > 0.0

    advanced = api._advance_gate_if_needed(np.array([0.0, 10.2, 1.5]))

    assert advanced
    assert api.last_gate_pass_preserved_plan
    assert api.current_gate_idx == 1
    assert api.active_waypoints is not None
    assert api.target_manager.locked
    assert api.active_target_track_id == -2
    np.testing.assert_allclose(api.current_gate_pos, np.array([0.0, 20.0, 1.5]))

import numpy as np

import autonomy_core.controller.attitude_controller3 as attitude_controller3
from autonomy_core.controller.attitude_controller3 import (
    RPGHighLevelTracker,
    Reference,
    State,
)


def test_near_reference_vertical_limit_caps_velocity_error_and_accel():
    tracker = RPGHighLevelTracker(
        kp=(0.0, 0.0, 0.0),
        kv=(0.0, 0.0, 10.0),
        max_acc_z_up=4.0,
        max_acc_z_down=4.0,
        max_acc_z_slew_m_s3=0.0,
        near_reference_z_error_m=0.5,
        near_reference_vz_error_max_m_s=0.1,
        near_reference_max_acc_z_up=0.5,
        near_reference_max_acc_z_down=0.5,
    )
    state = State(
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.array([0.0, 0.0, -10.0]),
        yaw=0.0,
    )
    ref = Reference(
        pos=np.array([0.0, 0.0, 0.1]),
        vel=np.zeros(3),
        acc=np.zeros(3),
        yaw=0.0,
    )

    _, _, _, _, debug = tracker.update(state, ref)

    assert debug["near_reference_vertical"]
    assert debug["vertical_velocity_error_limited"]
    np.testing.assert_allclose(debug["e_v_for_control"], [0.0, 0.0, 0.1])
    np.testing.assert_allclose(debug["a_cmd_accel_limited_no_g"], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(debug["a_cmd_no_g"], [0.0, 0.0, 0.5])


def test_vertical_accel_slew_limits_full_reversal(monkeypatch):
    times = iter((10.0, 10.1))
    monkeypatch.setattr(
        attitude_controller3.time,
        "monotonic",
        lambda: next(times),
    )
    tracker = RPGHighLevelTracker(
        kp=(0.0, 0.0, 0.0),
        kv=(0.0, 0.0, 0.0),
        max_acc_z_up=4.0,
        max_acc_z_down=4.0,
        max_acc_z_slew_m_s3=1.0,
        max_acc_z_slew_reset_s=1.0,
    )
    state = State(pos=np.zeros(3), vel=np.zeros(3), yaw=0.0)

    first_ref = Reference(
        pos=np.zeros(3),
        vel=np.zeros(3),
        acc=np.array([0.0, 0.0, -1.0]),
        yaw=0.0,
    )
    second_ref = Reference(
        pos=np.zeros(3),
        vel=np.zeros(3),
        acc=np.array([0.0, 0.0, 1.0]),
        yaw=0.0,
    )

    _, _, _, _, first_debug = tracker.update(state, first_ref)
    _, _, _, _, second_debug = tracker.update(state, second_ref)

    np.testing.assert_allclose(first_debug["a_cmd_no_g"], [0.0, 0.0, -1.0])
    assert second_debug["vertical_accel_slew_limited"]
    np.testing.assert_allclose(
        second_debug["a_cmd_accel_limited_no_g"],
        [0.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(second_debug["a_cmd_no_g"], [0.0, 0.0, -0.9])

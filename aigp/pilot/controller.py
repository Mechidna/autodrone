import time
import math

from pymavlink import mavutil
from hover_hold import HoverHold

# --------------------------------------------------------------------------------------
# RESET COMMAND
MAVLINK_CMD_SIM_RESET = 31000

# --------------------------------------------------------------------------------------
# MOTOR CONTROLS
# --------------------------------------------------------------------------------------

MOTOR_FRONT_LEFT = 0
MOTOR_FRONT_RIGHT = 1
MOTOR_BACK_LEFT = 0
MOTOR_BACK_RIGHT = 0

def update_motor_control(mavlink_conn, system_boot_ms):
    motor_rpms = [MOTOR_FRONT_LEFT, MOTOR_FRONT_RIGHT, MOTOR_BACK_LEFT, MOTOR_BACK_RIGHT, 0, 0, 0, 0]
    mavlink_conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        0,
        motor_rpms
    )

# --------------------------------------------------------------------------------------
# ATTITUDE CONTROLS
# --------------------------------------------------------------------------------------
def euler_to_quaternion(roll, pitch, yaw):
    """
    roll/pitch/yaw in radians.
    Returns MAVLink quaternion order: [w, x, y, z]
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]

def get_latest_autonomy_command(data):
    lock = data.get("lock") if isinstance(data, dict) else None

    if lock is not None:
        with lock:
            cmd = data.get("latest_autonomy_command")
            command_status = data.get("latest_autonomy_command_status", "unknown")
    else:
        cmd = data.get("latest_autonomy_command")
        command_status = data.get("latest_autonomy_command_status", "unknown")

    now = time.monotonic()

    if not hasattr(get_latest_autonomy_command, "last_cmd_print_time"):
        get_latest_autonomy_command.last_cmd_print_time = 0.0

    if now - get_latest_autonomy_command.last_cmd_print_time >= 1.0:
        if cmd is None:
            print(
                "controller command:",
                f"fallback hover status={command_status}",
                flush=True,
            )
        else:
            print(
                "controller command:",
                f"roll={cmd.roll_deg:.2f} deg",
                f"pitch={cmd.pitch_deg:.2f} deg",
                f"yaw={cmd.yaw_deg:.2f} deg",
                f"thrust={cmd.thrust:.3f}",
                flush=True,
            )
        get_latest_autonomy_command.last_cmd_print_time = now

    return cmd

def get_latest_attitude_command(data):
    cmd = get_latest_autonomy_command(data)

    if cmd is None:
        return 0.0, 0.0, 0.0, 0.74

    return cmd.roll_deg, cmd.pitch_deg, cmd.yaw_deg, cmd.thrust

def get_autonomy_attitude_command(data):
    return get_latest_attitude_command(data)

ANGLE_ATTITUDE_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE |
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE |
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
)

def send_attitude_angle_flight_control(mavlink_conn, system_boot_ms, roll_deg, pitch_deg, yaw_deg, thrust):
    now_ms = int(time.time() * 1000)

    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    q = euler_to_quaternion(roll, pitch, yaw)

    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        ANGLE_ATTITUDE_MASK,
        q,              # real attitude quaternion
        0.0, 0.0, 0.0,  # ignored body rates
        max(0.0, min(1.0, thrust))
    )

def update_attitude_angle_flight_control(mavlink_conn, system_boot_ms, data):
    roll_deg, pitch_deg, yaw_deg, thrust = get_latest_attitude_command(data)
    send_attitude_angle_flight_control(
        mavlink_conn,
        system_boot_ms,
        roll_deg,
        pitch_deg,
        yaw_deg,
        thrust,
    )

# PITCH_RATE = -0.3   # rad/s (negative = pitch forward)
# ROLL_RATE  = 0.0
# YAW_RATE   = 0.0
# THRUST     = 0.6    # 0.0 - 1.0
#
# RATES_ATTITUDE_MASK = (
#     mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
# )
# def update_attitude_flight_control(mavlink_conn, system_boot_ms):
#     now_ms = int(time.time() * 1000)
#
#     """
#     Sets a desired vehicle attitude. Used by an external controller to
#     command the vehicle (manual controller or other system).
#
#     time_boot_ms              : Timestamp (time since system boot). [ms] (type:uint32_t)
#     target_system             : System ID (type:uint8_t)
#     target_component          : Component ID (type:uint8_t)
#     type_mask                 : Bitmap to indicate which dimensions should be ignored by the vehicle. (type:uint8_t, values:ATTITUDE_TARGET_TYPEMASK)
#     q                         : Attitude quaternion (w, x, y, z order, zero-rotation is 1, 0, 0, 0) (type:float)
#     body_roll_rate            : Body roll rate [rad/s] (type:float)
#     body_pitch_rate           : Body pitch rate [rad/s] (type:float)
#     body_yaw_rate             : Body yaw rate [rad/s] (type:float)
#     thrust                    : Collective thrust, normalized to 0 .. 1 (-1 .. 1 for vehicles capable of reverse trust) (type:float)
#     """
#     mavlink_conn.mav.set_attitude_target_send(
#         now_ms - system_boot_ms,
#         mavlink_conn.target_system,
#         mavlink_conn.target_component,
#         RATES_ATTITUDE_MASK,
#         [1, 0, 0, 0],  # dummy quaternion (ignored)
#         ROLL_RATE,
#         PITCH_RATE,
#         YAW_RATE,
#         THRUST
#     )

# --------------------------------------------------------------------------------------
# POSITION CONTROLS
# --------------------------------------------------------------------------------------
VELOCITY_POSITION_MASK = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |

        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |

        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

def update_position_flight_control(mavlink_conn, system_boot_ms):
    now_ms = int(time.time() * 1000)

    """
    Sets a desired vehicle position in a local north-east-down coordinate
    frame. Used by an external controller to command the vehicle
    (manual controller or other system).

    time_boot_ms              : Timestamp (time since system boot). [ms] (type:uint32_t)
    target_system             : System ID (type:uint8_t)
    target_component          : Component ID (type:uint8_t)
    coordinate_frame          : Valid options are: MAV_FRAME_LOCAL_NED = 1, MAV_FRAME_LOCAL_OFFSET_NED = 7, MAV_FRAME_BODY_NED = 8, MAV_FRAME_BODY_OFFSET_NED = 9 (type:uint8_t, values:MAV_FRAME)
    type_mask                 : Bitmap to indicate which dimensions should be ignored by the vehicle. (type:uint16_t, values:POSITION_TARGET_TYPEMASK)
    x                         : X Position in NED frame [m] (type:float)
    y                         : Y Position in NED frame [m] (type:float)
    z                         : Z Position in NED frame (note, altitude is negative in NED) [m] (type:float)
    vx                        : X velocity in NED frame [m/s] (type:float)
    vy                        : Y velocity in NED frame [m/s] (type:float)
    vz                        : Z velocity in NED frame [m/s] (type:float)
    afx                       : X acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    afy                       : Y acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    afz                       : Z acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    yaw                       : yaw setpoint [rad] (type:float)
    yaw_rate                  : yaw rate setpoint [rad/s] (type:float)
    """
    mavlink_conn.mav.set_position_target_local_ned_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        VELOCITY_POSITION_MASK,
        0.0, 0, 0.0,    # ignored position NED
        2.0, 0.0, 0.0,  # Vel - 2 m/s forward
        0.0, 0, 0.0,    # ignored acceleration
        0,              # ignored yaw
        0.0             # ignored yaw rate
    )

# --------------------------------------------------------------------------------------
# Control Loop
# --------------------------------------------------------------------------------------

CONTROL_HZ = 50

class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.hover_hold = HoverHold()

    def update(self):
        cmd = get_latest_autonomy_command(self.data)

        if cmd is None:
            if not self.hover_hold.update_and_send(self.sim_conn, self.system_boot_ms, self.data):
                send_attitude_angle_flight_control(
                    self.sim_conn,
                    self.system_boot_ms,
                    0.0,
                    0.0,
                    0.0,
                    0.74,
                )
        else:
            self.hover_hold.reset()
            send_attitude_angle_flight_control(
                self.sim_conn,
                self.system_boot_ms,
                cmd.roll_deg,
                cmd.pitch_deg,
                cmd.yaw_deg,
                cmd.thrust,
            )

        # alternatively one of
        # update_position_flight_control(self.sim_conn, self.system_boot_ms)
        # update_motor_control(self.sim_conn, self.system_boot_ms)

        time.sleep(1.0 / CONTROL_HZ)

    # -------------------------------
    # Arm the drone
    # -------------------------------
    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,  # arm
            0, 0, 0, 0, 0, 0
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,  # confirmation
            0, 0, 0, 0, 0, 0, 0
        )

    # -------------------------------
    # Set Mode (px4)
    # -------------------------------
    def set_mode(self, mode_name):
        modes = self.sim_conn.mode_mapping()

        if modes is None or mode_name not in modes:
            raise RuntimeError(f"Mode {mode_name} not available")

        print(f"Setting mode: {mode_name}")
        self.sim_conn.set_mode(mode_name)

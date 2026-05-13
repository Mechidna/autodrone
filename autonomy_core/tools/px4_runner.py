# px4_runner.py
import asyncio
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from mavsdk import System
from mavsdk.offboard import Attitude, OffboardError

from autonomy_core.launch.get_telemetry import GetTelemetry
from autonomy_core.launch.autonomy_api6 import AutonomyAPI
from flight_logger import FlightLogger


# -------------------------------------------------
# Shared telemetry object
# -------------------------------------------------
telemetry = GetTelemetry()


def rad2deg(x):
    return x * 180.0 / math.pi


def hover_command(autonomy):
    yaw_rad = float(autonomy.telemetry.rpy["yaw"])
    return 0.0, 0.0, yaw_rad, autonomy.tracker.thrust_hover


# -------------------------------------------------
# ROS2 camera adapter
# -------------------------------------------------
class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_adapter_node")

        self.bridge = CvBridge()

        self.frame = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.last_frame_time = None

        camera_info_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            Image,
            "/camera",
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            "/camera_info",
            self.camera_info_callback,
            camera_info_qos,
        )

        self.get_logger().info("PerceptionNode started.")

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape(3, 3)

        d = np.array(msg.d, dtype=np.float32) if len(msg.d) > 0 else np.zeros((5,), dtype=np.float32)
        self.dist_coeffs = d.reshape(-1, 1)

    def image_callback(self, msg: Image):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_frame_time = time.time()
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")


def ros_spin_thread(node):
    rclpy.spin(node)


# -------------------------------------------------
# Telemetry tracking tasks
# -------------------------------------------------
async def track_position(drone):
    async for pv in drone.telemetry.position_velocity_ned():
        pos_xyz = (
            pv.position.north_m,
            pv.position.east_m,
            -pv.position.down_m,
        )
        telemetry.telemetry_pos(pos_xyz, start_position=None)


async def track_velocity(drone):
    async for pv in drone.telemetry.position_velocity_ned():
        vel_xyz = (
            pv.velocity.north_m_s,
            pv.velocity.east_m_s,
            -pv.velocity.down_m_s,
        )
        telemetry.telemetry_vel(vel_xyz, start_velocity=None)


async def track_orientation(drone):
    async for att in drone.telemetry.attitude_euler():
        rpy_rad = (
            math.radians(att.roll_deg),
            math.radians(att.pitch_deg),
            math.radians(att.yaw_deg),
        )
        telemetry.telemetry_rpy(rpy_rad, start_orientation=None)


# -------------------------------------------------
# Main
# -------------------------------------------------
async def main():
    # ---------------- ROS2 startup ----------------
    rclpy.init()
    perception_node = PerceptionNode()
    ros_thread = threading.Thread(target=ros_spin_thread, args=(perception_node,), daemon=True)
    ros_thread.start()

    # ---------------- MAVSDK startup ----------------
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone.")
            break

    use_perception = True
    autonomy = AutonomyAPI(use_perception=use_perception, race_gate_count=3)
    autonomy.telemetry = telemetry

    # -------------------------------------------------
    # Initial telemetry snapshot
    # -------------------------------------------------
    pv0 = await drone.telemetry.position_velocity_ned().__anext__()
    att0 = await drone.telemetry.attitude_euler().__anext__()

    start_position = (
        pv0.position.north_m,
        pv0.position.east_m,
        -pv0.position.down_m,
    )
    telemetry.telemetry_pos(start_position, start_position=start_position)

    start_velocity = (
        pv0.velocity.north_m_s,
        pv0.velocity.east_m_s,
        -pv0.velocity.down_m_s,
    )
    telemetry.telemetry_vel(start_velocity, start_velocity=start_velocity)

    start_orientation = (
        math.radians(att0.roll_deg),
        math.radians(att0.pitch_deg),
        math.radians(att0.yaw_deg),
    )
    telemetry.telemetry_rpy(start_orientation, start_orientation=start_orientation)

    # Start live telemetry tasks
    asyncio.create_task(track_position(drone))
    asyncio.create_task(track_velocity(drone))
    asyncio.create_task(track_orientation(drone))

    if use_perception:
        # Wait for ROS2 camera + camera_info only when perception is active.
        print("Waiting for camera frame and intrinsics...")
        while (
            perception_node.frame is None
            or perception_node.camera_matrix is None
            or perception_node.dist_coeffs is None
        ):
            await asyncio.sleep(0.05)

        print("Camera data ready.")

    # Let telemetry populate once
    await asyncio.sleep(0.2)

    if use_perception:
        # -------------------------------------------------
        # Warm up perception and memory BEFORE arming
        # -------------------------------------------------
        print("Warming up perception memory before arming...")
        warmup_deadline = time.time() + 1.0

        while time.time() < warmup_deadline:
            frame = perception_node.frame
            camera_matrix = perception_node.camera_matrix
            dist_coeffs = perception_node.dist_coeffs

            if frame is not None and camera_matrix is not None and dist_coeffs is not None:
                result = autonomy.update_gate_memory_from_frame(frame, camera_matrix, dist_coeffs)

                if result is not None and result.get("committed_now", False):
                    ok = autonomy.path_plan()
                    if ok:
                        print("Initial committed gate found; trajectory planned.")
                        break

            await asyncio.sleep(0.05)

        # If no committed gate yet, that's okay.
        # attitude_control() should fall back to hover-ish neutral command until a plan exists.
    else:
        print("Perception disabled; planning initial trajectory from mock GT gates.")
        ok = autonomy.path_plan()
        if not ok:
            print("Initial mock-gate planning failed; vehicle will hold neutral command.")

    # -------------------------------------------------
    # Arm only after perception warmup / initial planning attempt
    # -------------------------------------------------
    await drone.action.arm()

    # Compute first command
    roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = autonomy.attitude_control()

    print(
        f"Initial command: "
        f"roll={rad2deg(roll_cmd):.2f} deg, "
        f"pitch={rad2deg(pitch_cmd):.2f} deg, "
        f"yaw={rad2deg(yaw_cmd):.2f} deg, "
        f"thrust={thrust_cmd:.2f}"
    )

    # Send a few setpoints before starting offboard
    for _ in range(50):
        await drone.offboard.set_attitude(
            Attitude(
                roll_deg=rad2deg(roll_cmd),
                pitch_deg=rad2deg(pitch_cmd),
                yaw_deg=rad2deg(yaw_cmd),
                thrust_value=thrust_cmd,
            )
        )
        await asyncio.sleep(0.02)

    try:
        await drone.offboard.start()
        print("Offboard started.")
        flight_logger = FlightLogger("flight_log.csv")
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        await drone.action.disarm()
        perception_node.destroy_node()
        rclpy.shutdown()
        return

    try:
        last_loop_time = time.time()

        while True:
            loop_start = time.time()
            loop_dt = loop_start - last_loop_time
            last_loop_time = loop_start

            mem_result = None
            replan_requested = False
            replan_duration = 0.0
            hold_command = False
            stale_command_suppressed = False
            autonomy.last_perception_replan_trigger = False

            if loop_dt > 0.1:
                print(f"[WARN] control loop gap {loop_dt:.3f}s; suppressing aggressive command.")
                stale_command_suppressed = True

            if use_perception:
                frame = perception_node.frame
                camera_matrix = perception_node.camera_matrix
                dist_coeffs = perception_node.dist_coeffs

                if frame is None or camera_matrix is None or dist_coeffs is None:
                    print("Waiting for camera data...")
                    await asyncio.sleep(0.05)
                    continue

                # -------------------------------------------------
                # 1) Update gate memory from current frame
                # -------------------------------------------------
                mem_result = autonomy.update_gate_memory_from_frame(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                )

            # -------------------------------------------------
            # 2) Replan only when it makes sense
            # -------------------------------------------------
            should_replan = False

            # A new stable landmark gate was committed
            if mem_result is not None and mem_result.get("committed_now", False):
                print("New committed gate -> replanning.")
                should_replan = True
                if use_perception:
                    autonomy.last_perception_replan_trigger = True

            # Current active target passed
            gate_changed = autonomy.advance_gate_if_needed(threshold=1.0)
            if gate_changed:
                print("Active target advanced -> replanning.")
                should_replan = True
                if use_perception:
                    autonomy.last_perception_replan_trigger = True

            # Current trajectory finished
            if autonomy.planner.total_time > 0.0 and autonomy.time_elapsed >= autonomy.planner.total_time:
                print("Trajectory horizon exhausted -> replanning.")
                should_replan = True
                if use_perception:
                    autonomy.last_perception_replan_trigger = True

            # Avoid absurdly frequent replans, except perception gate completion.
            # Once a perceived landmark is marked complete, the old target must
            # be replaced immediately so it cannot be tracked again.
            force_perception_gate_replan = use_perception and gate_changed
            if should_replan and (
                force_perception_gate_replan
                or time.time() - autonomy.replan_time > 0.3
            ):
                replan_requested = True
                hold_command = True
                replan_started = time.time()
                print(f"[REPLAN] start t={replan_started:.3f}")
                ok = autonomy.path_plan()
                replan_duration = time.time() - replan_started
                print(
                    f"[REPLAN] end duration={replan_duration:.3f}s "
                    f"mode={autonomy.last_plan_mode} "
                    f"start_gate={autonomy.last_plan_start_gate_idx} "
                    f"end_gate={autonomy.last_plan_end_gate_idx}"
                )
                if not ok:
                    print("Replan requested, but no valid gates available yet.")

            # -------------------------------------------------
            # 3) Track current trajectory (or hover if none yet)
            # -------------------------------------------------
            if (
                (hold_command or stale_command_suppressed)
                and use_perception
                and len(getattr(autonomy, "active_target_gates", [])) == 0
            ):
                roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = autonomy.attitude_control()
            elif hold_command or stale_command_suppressed:
                roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = hover_command(autonomy)
            else:
                roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = autonomy.attitude_control()


            # flight logger
            p_ref = getattr(autonomy, "p_ref", None)
            v_ref = getattr(autonomy, "v_ref", None)
            a_ref = getattr(autonomy, "a_ref", None)

            target = getattr(autonomy, "current_gate", None)
            if target is None:
                target = getattr(autonomy, "current_target_gate", None)
            if target is None:
                target = getattr(autonomy, "target_gate", None)
            if target is None:
                target = getattr(autonomy, "active_gate", None)
            if target is None and use_perception:
                target = getattr(autonomy, "hold_anchor", None)

            mode = getattr(autonomy, "mode", None)

            active_gate_idx = getattr(autonomy, "active_gate_idx", None)
            if active_gate_idx is None:
                active_gate_idx = getattr(autonomy, "current_gate_idx", None)
            if active_gate_idx is None:
                active_gate_idx = getattr(autonomy, "target_gate_idx", None)

            flight_logger.log(
                telemetry=telemetry,
                roll_cmd=roll_cmd,
                pitch_cmd=pitch_cmd,
                yaw_cmd=yaw_cmd,
                thrust_cmd=thrust_cmd,
                p_ref=p_ref,
                v_ref=v_ref,
                a_ref=a_ref,
                target=target,
                mode=mode,
                active_gate_idx=active_gate_idx,
                loop_dt=loop_dt,
                replan_requested=replan_requested,
                replan_duration=replan_duration,
                hold_command=hold_command,
                stale_command_suppressed=stale_command_suppressed,
                plan_mode=getattr(autonomy, "last_plan_mode", None),
                plan_start_gate_idx=getattr(autonomy, "last_plan_start_gate_idx", None),
                plan_end_gate_idx=getattr(autonomy, "last_plan_end_gate_idx", None),
                raw_gate=getattr(autonomy, "last_raw_gate_center", None),
                perception_accepted=getattr(autonomy, "last_perception_accepted", False),
                perception_rejection_reason=getattr(autonomy, "last_perception_rejection_reason", ""),
                last_valid_target=getattr(autonomy, "last_valid_target", None),
                target_z_clamped=getattr(autonomy, "last_target_z_clamped", False),
                perception_replan_trigger=getattr(autonomy, "last_perception_replan_trigger", False),
                distance_to_active_target=getattr(autonomy, "distance_to_active_target", float("nan")),
                gate_completion_triggered=getattr(autonomy, "gate_completion_triggered", False),
                completion_reason=getattr(autonomy, "completion_reason", ""),
                completed_gate_position=getattr(autonomy, "completed_gate_position", None),
                active_gate_idx_before=getattr(autonomy, "active_gate_idx_before", None),
                active_gate_idx_after=getattr(autonomy, "active_gate_idx_after", None),
                race_cursor_before=getattr(autonomy, "race_cursor_before", None),
                race_cursor_after=getattr(autonomy, "race_cursor_after", None),
                active_target_source=getattr(autonomy, "active_target_source", ""),
                target_rejected_completed=getattr(autonomy, "target_rejected_completed", False),
                candidate_track_id=getattr(autonomy, "candidate_track_id", None),
                candidate_center=getattr(autonomy, "candidate_center", None),
                candidate_order_score=getattr(autonomy, "candidate_order_score", float("nan")),
                rejected_wrong_order=getattr(autonomy, "rejected_wrong_order", False),
                rejected_duplicate=getattr(autonomy, "rejected_duplicate", False),
                rejected_completed_this_lap=getattr(autonomy, "rejected_completed_this_lap", False),
                race_cursor_advanced=getattr(autonomy, "race_cursor_advanced", False),
                active_gate_idx_advanced=getattr(autonomy, "active_gate_idx_advanced", False),
                completed_landmark_count=getattr(autonomy, "completed_landmark_count", 0),
                lap_reset_triggered=getattr(autonomy, "lap_reset_triggered", False),
                active_target_cleared=getattr(autonomy, "active_target_cleared", False),
                active_target_track_id=getattr(autonomy, "active_target_track_id", None),
                completed_gate_track_id=getattr(autonomy, "completed_gate_track_id", None),
                yaw_target_source=getattr(autonomy, "yaw_target_source", ""),
                target_retained_after_completion=getattr(autonomy, "target_retained_after_completion", False),
                next_valid_target_found=getattr(autonomy, "next_valid_target_found", False),
                valid_candidate_count=getattr(autonomy, "valid_candidate_count", 0),
                approach_vector=getattr(autonomy, "approach_vector", None),
                gate_progress_along_approach=getattr(autonomy, "gate_progress_along_approach", float("nan")),
                gate_lateral_error=getattr(autonomy, "gate_lateral_error", float("nan")),
                gate_plane_crossed=getattr(autonomy, "gate_plane_crossed", False),
                near_gate_but_not_crossed=getattr(autonomy, "near_gate_but_not_crossed", False),
                completion_blocked_reason=getattr(autonomy, "completion_blocked_reason", ""),
                no_active_target=getattr(autonomy, "no_active_target", False),
                no_target_control_mode=getattr(autonomy, "no_target_control_mode", ""),
                hold_anchor_source=getattr(autonomy, "hold_anchor_source", ""),
                hold_anchor=getattr(autonomy, "hold_anchor", None),
                velocity_damping_active=getattr(autonomy, "velocity_damping_active", False),
                completed_gate_reference_blocked=getattr(autonomy, "completed_gate_reference_blocked", False),
                p_ref_source=getattr(autonomy, "p_ref_source", ""),
                yaw_hold_value=math.degrees(getattr(autonomy, "yaw_hold_value", float("nan"))),
                telemetry_yaw_deg=float(getattr(autonomy.telemetry, "rpy", {}).get("yaw", float("nan"))),
                previous_yaw_cmd_deg=math.degrees(getattr(autonomy, "previous_yaw_cmd_log", float("nan"))),
                raw_yaw_cmd_deg=math.degrees(getattr(autonomy, "raw_yaw_cmd", float("nan"))),
                yaw_cmd_after_unwrap_deg=math.degrees(getattr(autonomy, "yaw_cmd_after_unwrap", float("nan"))),
                yaw_rate_limited=getattr(autonomy, "yaw_rate_limited", False),
                post_completion_grace_active=getattr(autonomy, "post_completion_grace_active", False),
                no_target_roll_source=getattr(autonomy, "no_target_roll_source", ""),
                no_target_pitch_source=getattr(autonomy, "no_target_pitch_source", ""),
                horizontal_hold_disabled_after_completion=getattr(autonomy, "horizontal_hold_disabled_after_completion", False),
                track_id=getattr(autonomy, "track_id", None),
                merged_into_track_id=getattr(autonomy, "merged_into_track_id", None),
                duplicate_merge_reason=getattr(autonomy, "duplicate_merge_reason", ""),
                race_order_track_ids=getattr(autonomy, "race_order_track_ids", []),
                race_order_inserted=getattr(autonomy, "race_order_inserted", False),
                race_order_rejected_reason=getattr(autonomy, "race_order_rejected_reason", ""),
                landmark_uncertainty=getattr(autonomy, "landmark_uncertainty", float("nan")),
                track_observations=getattr(autonomy, "track_observations", 0),
                completed_unique_gate_count=getattr(autonomy, "completed_unique_gate_count", 0),
                active_gate_idx_clamped_by_race_gate_count=getattr(autonomy, "active_gate_idx_clamped_by_race_gate_count", False),
                suspected_duplicate_track=getattr(autonomy, "suspected_duplicate_track", False),
                committed_track_centers=getattr(autonomy, "committed_track_centers_log", ""),
                pairwise_committed_track_distances=getattr(autonomy, "pairwise_committed_track_distances", ""),
                duplicate_radius_used=getattr(autonomy, "duplicate_radius_used", float("nan")),
                merge_candidate_pairs=getattr(autonomy, "merge_candidate_pairs", ""),
                merge_blocked_reason=getattr(autonomy, "merge_blocked_reason", ""),
                rejected_track_temporary_vs_permanent=getattr(autonomy, "rejected_track_temporary_vs_permanent", ""),
                active_target_admission_status=getattr(autonomy, "active_target_admission_status", ""),
                race_order_after_merge=getattr(autonomy, "race_order_after_merge", []),
            )

            await drone.offboard.set_attitude(
                Attitude(
                    roll_deg=rad2deg(roll_cmd),
                    pitch_deg=rad2deg(pitch_cmd),
                    yaw_deg=rad2deg(yaw_cmd),
                    thrust_value=thrust_cmd,
                )
            )

            print(
                f"cmd | roll={rad2deg(roll_cmd):6.2f} deg "
                f"pitch={rad2deg(pitch_cmd):6.2f} deg "
                f"yaw={rad2deg(yaw_cmd):7.2f} deg "
                f"thrust={thrust_cmd:4.2f}"
            )

            await asyncio.sleep(0.02)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        try:
            flight_logger.close()
        except Exception:
            pass

        try:
            await drone.offboard.stop()
        except Exception:
            pass

        try:
            await drone.action.land()
        except Exception:
            pass

        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

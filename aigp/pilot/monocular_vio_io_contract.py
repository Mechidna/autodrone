# Monocular VIO Inputs From This Stack
#
# - Camera image frame from UDP competition vision or ROS camera receiver.
# - Camera frame timestamp from the image packet/header when available.
# - Camera intrinsics: fx, fy, cx, cy, image width, image height.
# - Camera distortion coefficients.
# - Camera-to-body / camera-to-IMU extrinsic rotation.
# - Camera-to-body / camera-to-IMU extrinsic translation.
# - Camera mount profile and yaw correction.
# - HIGHRES_IMU accelerometer samples: xacc, yacc, zacc.
# - HIGHRES_IMU gyroscope samples: xgyro, ygyro, zgyro.
# - HIGHRES_IMU timestamp: time_usec.
# - IMU sample receive wall time as fallback timing/debug metadata.
# - TIMESYNC data when available for simulator/client clock alignment.
# - HEARTBEAT armed state for estimator mode gating/reset policy.
# - ACTUATOR_OUTPUT_STATUS motor outputs as optional excitation/debug metadata.
# - ENCAPSULATED_DATA race status as optional run-context metadata.
# - ENCAPSULATED_DATA track/gate info as optional evaluation/context metadata.
# - Runtime configuration for VIO feature tracking limits.
# - Runtime configuration for IMU noise, gravity, and bias assumptions.
# - Runtime configuration for estimator initialization mode.
# - Runtime configuration for estimator maximum state age/freshness.
# - Optional known gate positions only for evaluation or external correction, not pure monocular VIO.
# - Optional latest YOLO/PnP gate observations only for landmark correction, not generic monocular VIO.
#
# Cross-Platform VIO Process Boundary
#
# - External VIO implementations should speak localhost TCP with one compact JSON
#   request or response per newline. This keeps the interface usable from Linux,
#   Windows, WSL, or a separate process/language runtime.
# - Image payloads are JPEG bytes encoded as base64 in JSON.
# - IMU payloads must include a non-empty, strictly increasing time_usec sequence
#   with finite accel_xyz and gyro_xyz values.
# - Do not expose Unix domain sockets, POSIX paths, shared-memory handles, or
#   platform-specific process control through the protocol. Those belong behind
#   the implementation-specific client/server launcher.
#
# Standard Professional Monocular VIO Outputs
#
# - Output timestamp.
# - Estimator status: initializing, tracking, degraded, lost, reset.
# - World/local position estimate.
# - World/local orientation estimate as quaternion.
# - World/local orientation estimate as roll, pitch, yaw when needed by downstream consumers.
# - World/local linear velocity estimate.
# - Body-frame angular velocity estimate.
# - Body-frame specific force or bias-corrected acceleration estimate.
# - Gravity direction estimate.
# - Gyroscope bias estimate.
# - Accelerometer bias estimate.
# - Camera-to-IMU extrinsic estimate or calibrated extrinsic reference.
# - Camera/IMU time-offset estimate or calibrated time-offset reference.
# - Pose covariance or uncertainty.
# - Velocity covariance or uncertainty.
# - Bias covariance or uncertainty.
# - Tracked feature count.
# - Inlier feature count.
# - Outlier feature count.
# - Keyframe count in the active sliding window.
# - Visual reprojection residual statistics.
# - IMU residual statistics.
# - Estimated feature depths or inverse depths for active visual landmarks.
# - Feature depth uncertainty or triangulation quality.
# - Scale estimate and scale observability status.
# - Initialization confidence.
# - Tracking quality score.
# - Failure reason when tracking is lost.
# - Reset counter.
# - Map/keyframe debug data when the implementation exposes it.
# - Local odometry transform suitable for control/planning consumers.
# - Freshness/latency metadata for control safety checks.

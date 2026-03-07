# 1. PREDICT (High frequency - IMU)
# Use acceleration and RPY to guess the new position
state_prediction = physics_model(current_state, imu_data)
uncertainty += imu_drift_noise

# 2. FEATURE MATCHING (Lower frequency - Camera)
# Find landmarks, use Camera Matrix to find relative motion
visual_delta_pose = visual_odometry(current_frame, prev_frame, K_matrix)

# 3. UPDATE (The Meeting Point)
# Calculate the 'Kalman Gain' (Who do we trust more?)
K_gain = uncertainty / (uncertainty + sensor_noise)

# Correct the state: New State = Prediction + (Gain * Difference)
final_state = state_prediction + K_gain * (visual_delta_pose - state_prediction)
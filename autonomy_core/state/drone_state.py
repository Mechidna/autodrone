# Logic inside your Decision Loop
dist = np.linalg.norm(gate_pos - drone_pos)

if dist > 5.0:
    # --- REFLEX MODE ---
    # Target: The gate. Velocity: Keep moving forward.
    target_vel = (gate_pos - drone_pos) / dist * 2.0 # 2m/s approach
    planner.update(current_ekf, {'pos': gate_pos, 'vel': target_vel}, duration=3.0)

elif dist <= 5.0 and confidence > 0.8:
    # --- LOGIC MODE ---
    # Target: 1 meter BEHIND the gate to ensure we fly THROUGH it.
    # Velocity: High, aligned with the gate's normal vector.
    pass_through_point = gate_pos + (gate_normal * 2.0)
    planner.update(current_ekf, {'pos': pass_through_point, 'vel': gate_normal * 5.0}, duration=1.5)
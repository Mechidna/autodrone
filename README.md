# autodrone

W.I.P. autonomous drone stack for PX4 SITL and MAVSDK in Python focused on autonomous drone racing/navigation.
Trajectory generation and control concepts are partially inspired by published work from UZH RPG and related autonomous drone racing research.

- ROS 2 Jazzy required for perception tools
- Currently tested with Gazebo Sim + PX4 SITL
- Example launch:
  `PX4_GZ_WORLD=gate_test make px4_sitl gz_x500_mono_cam`
- Primary stack: Python, MAVSDK, PX4, Gazebo, ROS 2

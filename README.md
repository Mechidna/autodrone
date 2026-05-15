# autodrone

W.I.P. autonomous drone stack for PX4 SITL and MAVSDK in Python focused on autonomous drone racing/navigation.
Trajectory generation and control concepts are partially inspired by published work from UZH RPG and related autonomous drone racing research.
Perception utilizes OpenCV-based image processing: HSV color thresholding, contour extraction, quadrilateral/corner ordering, and planar pose estimation with OpenCV solvePnP using IPPE/ITERATIVE PnP variants.

- ROS 2 Jazzy required for establishing camera node to Gazebo
- Currently tested with Gazebo Sim + PX4 SITL
- Example launch:
  `PX4_GZ_WORLD=gate_test make px4_sitl gz_x500_mono_cam`
- Primary stack: Python, MAVSDK, PX4, Gazebo, ROS 2

<img width="640" height="480" alt="ezgif com-resize" src="https://github.com/user-attachments/assets/8df84775-5189-4fea-b656-7c6eccf73df6" />

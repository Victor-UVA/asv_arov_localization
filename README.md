# asv_arov_localization
ROS2 package to run an EKF to estimate the AROV's position globally using AprilTags in known, fixed positions.
```
ros2 run asv_arov_localization arov_ekf_global --ros-args --remap __ns:=/arov
```

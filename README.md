# asv_arov_localization
ROS2 package to run an EKF to estimate the AROV's position globally using AprilTags in known, fixed positions.
```
ros2 run asv_arov_localization arov_ekf_global --ros-args --remap __ns:=/arov
```

## Getting PCL installed on jazzy
Clean and update first:
```
sudo apt clean
sudo apt update
apt-cache policy ros-jazzy-pcl-ros
```
If the previous command returns 2.6.2-2 or 2.6.3-0:
```
sudo apt install ros-jazzy-pcl-ros
```
Check the package executables were installed correctly, should see a list of them:
```
ros2 pkg executables pcl_ros
```

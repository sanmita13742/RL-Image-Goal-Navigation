#!/bin/bash

# terminal 1
gnome-terminal --title="Ranger Bringup" -- bash -c "
source /opt/ros/foxy/setup.bash
source ~/agilex_ws/install/setup.bash

cd ~/agilex_ws/src/ranger_ros2/ranger_bringup/scripts
./bringup_can2usb.bash

ros2 launch ranger_bringup ranger_mini_v3.launch.py

exec bash
"

# Wait for bringup
sleep 5

# terminal 2
gnome-terminal --title="Keyboard Teleop" -- bash -c "
source /opt/ros/foxy/setup.bash
source ~/agilex_ws/install/setup.bash

ros2 run teleop_twist_keyboard teleop_twist_keyboard

exec bash
"

#for gui - ros2 run ranger_teleop_gui teleop_gui
sleep 2

# terminal 3
gnome-terminal --title="RealSense Camera" -- bash -c "
source /opt/ros/foxy/setup.bash
source ~/agilex_ws/install/setup.bash

ros2 launch realsense2_camera rs_launch.py

exec bash
"

sleep 2

# terminal 4
gnome-terminal --title="RoboSense LiDAR" -- bash -c "
source /opt/ros/foxy/setup.bash
source ~/rslidar_ws/install/setup.bash
source ~/agilex_ws/install/setup.bash

ros2 launch ranger_bringup open_rslidar.launch.py

exec bash
"

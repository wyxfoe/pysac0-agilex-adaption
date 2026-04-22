#!/bin/bash
source ~/.bashrc

workspace=$(pwd)
# /home/razer/桌面/Ecat/ecat_follow_control_source/src/follow_control.sh
gnome-terminal -t "follow1" -x sudo bash -c "cd workspace;source ./devel/setup.bash && roslaunch arm_control arx5.launch; exec bash;"
sleep 1
# gnome-terminal -t "follow2" -x sudo bash -c "cd /home/razer/桌面/Ecat/ecat_arx_follow_source/follow2;source ./devel/setup.bash && roslaunch arm_control arx5v.launch; exec bash;"
# sleep 1


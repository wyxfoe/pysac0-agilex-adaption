#!/bin/bash
workspace=$(pwd)

gnome-terminal -t "follow1" -- sudo bash -c "cd ${workspace}/follow1; source ./devel/setup.bash; roslaunch arm_control arx.launch; exec bash;"
sleep 1
gnome-terminal -t "follow2" -- sudo bash -c "cd ${workspace}/follow2; source ./devel/setup.bash; roslaunch arm_control arx.launch; exec bash;"
sleep 1

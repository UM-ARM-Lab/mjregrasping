# MJRegrasping


## Install

Create a catkin workspace and clone the repository:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone git@github.com:UM-ARM-Lab/mjregrasping.git
```

Install dependencies:

```bash
# ROS dependencies
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
# Create a virtual environment with access to the system ROS packages
cd ~/catkin_ws
# If you're using Ubuntu 20.04 and do not have python3.10 installed,
# you can add it via the deadsnakes PPA
sudo apt install python3.10-venv
python3.10 -m venv --system-site-packages venv
source venv/bin/activate
# Python dependencies
cd ~/catkin_ws/src/mjregrasping
pip install -r requirements.txt
```

Build the workspace:

```bash
cd ~/catkin_ws
catkin build
source devel/setup.bash
```

## Running the demo

```bash
roslaunch mjregrasping demo.launch
```

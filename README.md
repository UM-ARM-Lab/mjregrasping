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
venv --system-site-packages venv
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

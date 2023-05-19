# MJRegrasping


## Install

Create a catkin workspace and clone the repository:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone git@github.com:UM-ARM-Lab/mjregrasping.git
wstool init
cp mjregrasping/mjregrasping.rosinstall .rosinstall
wstool update
```

Install dependencies:

```bash
# ROS dependencies
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
# Create a virtual environment with access to the system ROS packages
cd ~/catkin_ws
# you may need sudo apt install python3.8-venv
python3 -m venv --system-site-packages venv
source venv/bin/activate
# Python dependencies
cd ~/catkin_ws/src/mjregrasping
pip install --upgrade pip
pip install --ignore-installed -r requirements.txt  # use ignore-installed because we want to override things like numpy/scipy that may be installed system-wide
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

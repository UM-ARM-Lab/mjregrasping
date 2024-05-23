# mjregrasping

This code can be used in two ways, with ROS or without ROS.
The project was originally developed on top of ROS1 Noetic, and all experiments from the ICRA paper depend on that.
The ROS stack depends on internal ARMLab packages like `hdt_description`, so unless you're trying to reproduce our experiments, you probably don't want to use this method.

For accesibility and ease of use, we also provide a non-ROS version of the code.
This is the recommended way if all you want to do is compute the h-signature or GL-signature for your own system.
It also provides a demo with a simple built-in robot described by MJCF (MuJoCo).

## Installation

Create a virtual environment and install the requirements:

```bash
python3 -m venv venv  # or however you prefer to create virtual environments
source venv/bin/activate
pip install -r requirements.txt
```

## Demos

### Simple Paths Demo

Run `scripts/homotopy_demo.py`

This uses hard-coded grasp and obstacle loops, and visualizes in [rerun](https://rerun.io/).
It should look like this:

![Simple Paths Demo](docs/homotopy_demo_rerun.png)

### Robot Grasping Cable Demo

Run `scripts/robot_grasping_cable_demo.py`

This uses the state of the mujoco world, and also visualizes the grasp loops and obstacles loops in rerun.
To see what it should look like in rerun, see the video: [robot_grasping_cable_demo.webm](docs/robot_grasping_cable_demo.webm)
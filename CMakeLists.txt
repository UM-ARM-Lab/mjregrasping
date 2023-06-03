cmake_minimum_required(VERSION 3.0)

project(mjregrasping)

find_package(catkin REQUIRED COMPONENTS rospy dynamic_reconfigure)

catkin_python_setup()

generate_dynamic_reconfigure_options(cfg/Params.cfg)

catkin_package(
        LIBRARIES mjregrasping
        CATKIN_DEPENDS ${CATKIN_DEPENDS}
)


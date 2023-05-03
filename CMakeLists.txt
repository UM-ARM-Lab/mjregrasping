cmake_minimum_required(VERSION 3.0)

project(mjregrasping)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES mjregrasping
        CATKIN_DEPENDS ${CATKIN_DEPENDS}
)

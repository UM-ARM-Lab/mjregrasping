cmake_minimum_required(VERSION 3.0)

project(mjregrasping)

find_package(catkin REQUIRED COMPONENTS rospy)

set(CMAKE_CXX_STANDARD 20)

catkin_python_setup()

catkin_package()
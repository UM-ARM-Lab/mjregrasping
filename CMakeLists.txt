cmake_minimum_required(VERSION 3.0)

project(mjregrasping)

find_package(catkin REQUIRED COMPONENTS rospy)
find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)

set(MUJOCO_HOME "$ENV{VIRTUAL_ENV}/lib/python3.8/site-packages/mujoco")
find_library(MUJOCO_LIBRARY libmujoco.so.2.3.3 ${MUJOCO_HOME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(THREADS_PREFER_PTHREAD_FLAG ON)

catkin_python_setup()

catkin_package()

include_directories(include ${MUJOCO_HOME}/include)
add_library(mjregrasping SHARED src/mjregrasping.cpp)

add_executable(perf_rollout src/perf_rollout.cpp)
target_link_directories(perf_rollout PUBLIC ${MUJOCO_HOME})
target_link_libraries(perf_rollout PUBLIC mjregrasping Threads::Threads ${MUJOCO_LIBRARY})

pybind11_add_module(pymjregrasping src/bindings.cpp)
target_link_libraries(pymjregrasping PUBLIC mjregrasping)

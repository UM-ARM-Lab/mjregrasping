cmake_minimum_required(VERSION 3.0)

project(mjregrasping)

find_package(catkin REQUIRED COMPONENTS rospy)
find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)

set(MUJOCO_HOME "$ENV{VIRTUAL_ENV}/lib/python3.8/site-packages/mujoco")
find_library(MUJOCO_LIBRARY libmujoco.so.2.3.3 ${MUJOCO_HOME})

set(CMAKE_CXX_STANDARD 20)
set(THREADS_PREFER_PTHREAD_FLAG ON)

catkin_python_setup()

catkin_package()

# pybind11_add_module(pymujoco_mppi src/bindings.cpp)
# target_include_directories(pymujoco_mppi PUBLIC include)
# target_link_libraries(pymujoco_mppi PUBLIC mujoco_mppi)

add_executable(perf_rollout src/perf_rollout.cpp)
target_include_directories(perf_rollout PUBLIC include ${MUJOCO_HOME}/include)
target_link_directories(perf_rollout PUBLIC ${MUJOCO_HOME})
target_link_libraries(perf_rollout PUBLIC Threads::Threads ${MUJOCO_LIBRARY})


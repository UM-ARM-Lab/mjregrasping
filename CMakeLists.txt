cmake_minimum_required(VERSION 3.0)

project(mjregrasping)

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(MUJOCO_BUILD_EXAMPLES ON)
set(MUJOCO_BUILD_SIMULATE ON)
set(MUJOCO_TEST_PYTHON_UTIL ON)
set(MUJOCO_BUILD_TESTS OFF)

add_subdirectory(dependencies)

add_library(mujoco_mppi src/mujoco_mppi.cpp)
target_include_directories(mujoco_mppi PUBLIC include)
target_link_libraries(mujoco_mppi
    PUBLIC
    mujoco::mujoco
    mujoco::libsimulate
    )

pybind11_add_module(pymujoco_mppi src/bindings.cpp)
target_include_directories(pymujoco_mppi PUBLIC include)
target_link_libraries(pymujoco_mppi PUBLIC mujoco_mppi)

add_executable(perf_rollout src/perf_rollout.cpp)
target_include_directories(perf_rollout PUBLIC include)
target_link_libraries(perf_rollout
    PUBLIC
    Threads::Threads
    mujoco::mujoco
    mujoco::libsimulate
    )
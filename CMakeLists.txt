cmake_minimum_required(VERSION 3.0)

project(mjregrasping)

set(CMAKE_CXX_STANDARD 20)

set(MUJOCO_BUILD_EXAMPLES OFF)
set(MUJOCO_BUILD_SIMULATE ON)
set(MUJOCO_BUILD_TESTS OFF)
set(MUJOCO_TEST_PYTHON_UTIL OFF)
set(DABSL_BUILD_TESTING OFF)

add_subdirectory(dependencies)

add_executable(app src/app.cpp)
target_include_directories(app PUBLIC include)
target_link_libraries(app
    mujoco::mujoco
    mujoco::libsimulate
    elasticity
    glfw
    absl::flags
    absl::flags_parse)



#include <unistd.h>
#include <iostream>

#include <mujoco/mujoco.h>

#include <mujoco_mppi/mujoco_mppi.hpp>

MujocoMPPI::MujocoMPPI(mjModel *model, mjData *data) : model_(model), data_(data)
{
}
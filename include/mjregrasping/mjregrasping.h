#pragma once

#include <vector>
#include <mujoco/mujoco.h>
#include "ctpl_stl.h"

using State = std::vector<double>;
using Control = std::vector<double>;

std::vector<State> rollout_one_trajectory(mjModel const *model, mjData *data, std::vector<Control> const &controls);

void parallel_rollout(mjModel const *model, mjData *data, std::vector<mjData *> const &datas,
                      std::vector<std::vector<Control>> const &controls);

std::vector<mjData *> preallocate_data_for_threads(mjModel const *model, int n_samples);

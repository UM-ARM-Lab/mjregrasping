#include <vector>
#include <iostream>
#include <random>
#include <cmath>

#include <mjregrasping/mjregrasping.h>

auto const N_SUB_TIME = 50;

std::vector<State>
rollout_one_trajectory(mjModel const *const model, mjData *data, std::vector<Control> const &controls) {
    std::vector<State> qs;
    for (auto const &control_t: controls) {
        std::copy(control_t.begin(), control_t.end(), data->ctrl);
        for (int sub_t = 0; sub_t < N_SUB_TIME; sub_t++) {
            mj_step(model, data);
        }
        qs.emplace_back(State(data->qpos, data->qpos + model->nq));
    }
    return qs;
}

void parallel_rollout(mjModel const *model, mjData *data, std::vector<mjData *> const &datas,
                      std::vector<std::vector<Control>> const &controls) {
    std::vector<std::future<std::vector<State>>> results;
    for (auto i = 0; i < controls.size(); ++i) {
        auto const &controls_i = controls[i];

        auto *data_i = datas[i];
        mj_copyData(data_i, model, data);

        results.emplace_back(std::async(
                std::launch::async,
                [&model, data_i, &controls_i]() {
                    return rollout_one_trajectory(model, data_i, controls_i);
                }));
    }

    // this effectively waits for all the futures to be ready
    for (auto &result: results) {
        result.get();
    }
}

std::vector<mjData *> preallocate_data_for_threads(mjModel const *const model, int const n_samples) {
    std::vector<mjData *> datas;
    for (auto i = 0; i < n_samples; ++i) {
        auto *empty_data = mj_makeData(model);
        datas.emplace_back(empty_data);
    }
    return datas;
}
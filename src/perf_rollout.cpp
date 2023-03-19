#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mujoco/mujoco.h>
#include <ctpl_stl.h>

auto const N_TIME = 10;
auto const N_SUB_TIME = 20;

using State = std::vector<double>;
using Control = std::vector<double>;

std::vector<State> rollout_one_trajectory(mjModel const *model, mjData *data, std::vector<Control> const &controls)
{
    std::vector<State> qs;
    for (int t = 0; t < N_TIME; t++)
    {
        Control control_t = controls[t];
        std::copy(control_t.begin(), control_t.end(), data->ctrl);
        for (int sub_t = 0; sub_t < N_SUB_TIME; sub_t++)
        {
            mj_step(model, data);
        }
        qs.emplace_back(State(data->qpos, data->qpos + model->nq));
    }
    return qs;
}

int main()
{
    // Create a model
    auto constexpr errstr_sz = 1000;
    char errstr[errstr_sz];
    errstr[errstr_sz - 1] = '\0'; // ensure null termination
    auto *model = mj_loadXML("val_scene.xml", nullptr, errstr, errstr_sz);

    if (!model)
    {
        std::cerr << "Error loading model: " << errstr << std::endl;
        return EXIT_FAILURE;
    }

    auto const n_threads = static_cast<int>(std::thread::hardware_concurrency());
    std::cout << "n_threads: " << n_threads << std::endl;
    ctpl::thread_pool p{n_threads};

    for (auto const n_samples : std::vector<int>{1, 4, 8, 80, 250, 500})
    {
        // Create a 2D vector of controls of shape [N_TIME, N_CTRL]
        std::vector<std::vector<Control>> controls;
        for (auto i = 0; i < n_samples; ++i)
        {
            std::vector<Control> controls_i;
            for (auto j = 0; j < N_TIME; ++j)
            {
                Control control_t(model->nu, 0.0);
                controls_i.emplace_back(control_t);
            }
            controls.emplace_back(controls_i);
        }

        auto const t0 = std::chrono::high_resolution_clock::now();

        std::vector<std::future<std::vector<State>>> results;
        for (int i = 0; i < n_samples; ++i)
        {
            // copy of data and model for the thread
            auto const &controls_i = controls[i];
            results.emplace_back(p.push(
                [&](int)
                {
                    auto *data = mj_makeData(model);
                    return rollout_one_trajectory(model, data, controls_i);
                }));
            // rollout_one_trajectory(model, data, controls[i]);
        }

        // this effectively waits for all the futures to be ready
        for (auto &result : results)
        {
            result.get();
        }

        auto const dt = std::chrono::high_resolution_clock::now() - t0;
        // print dt in seconds as a decimal number
        auto const dt_s = std::chrono::duration<double>(dt).count();
        std::cout << std::setw(4) << " | " << n_samples << " | " << dt_s << " | " << std::endl;
    }

    return EXIT_SUCCESS;
}
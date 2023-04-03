#include <iostream>
#include <iomanip>

#include <mjregrasping/mjregrasping.h>

auto const N_TIME = 10;

int main(int argc, char *argv[]) {
    // Create a model
    auto constexpr errstr_sz = 1000;
    char errstr[errstr_sz];
    errstr[errstr_sz - 1] = '\0'; // ensure null termination
    auto *model = mj_loadXML(argv[1], nullptr, errstr, errstr_sz);

    if (!model) {
        std::cerr << "Error loading model: " << errstr << std::endl;
        return EXIT_FAILURE;
    }

    auto const n_threads = static_cast<int>(std::thread::hardware_concurrency());
    std::cout << "n_threads: " << n_threads << std::endl;
    ctpl::thread_pool p{n_threads};
    auto *data = mj_makeData(model);

    int const n_samples = 50;

    // Create a 2D vector of controls of shape [N_TIME, N_CTRL]
    std::vector<std::vector<Control>> controls;
    std::vector<mjData *> datas;
    for (auto i = 0; i < n_samples; ++i) {
        std::vector<Control> controls_i;
        for (auto j = 0; j < N_TIME; ++j) {
            Control control_t(model->nu, 0.0);
            controls_i.emplace_back(control_t);
        }
        controls.emplace_back(controls_i);

        auto *empty_data = mj_makeData(model);
        datas.emplace_back(empty_data);
    }

    for (auto i = 0; i < 10; ++i) {
        auto const t0 = std::chrono::high_resolution_clock::now();

        parallel_rollout(model, data, datas, controls);

        auto const dt = std::chrono::high_resolution_clock::now() - t0;
        // print dt in seconds as a decimal number
        auto const dt_s = std::chrono::duration<double>(dt).count();
        std::cout << std::setw(4) << " | " << n_samples << " | " << dt_s << " | " << std::endl;
    }

    return EXIT_SUCCESS;
}
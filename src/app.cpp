#include <absl/flags/flag.h>
mm
#include <absl/flags/parse.h>
#include <absl/strings/match.h>
#include <mujoco/mujoco.h>
#include <simulate.h>
#include <glfw_adapter.h>

int main(int argc, char **argv)
{
    absl::ParseCommandLine(argc, argv);

    std::printf("MuJoCo version %s\n", mj_versionString());
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have Different versions");
    }

    // threads
    auto sim = std::make_unique<mujoco::Simulate>(std::make_unique<mujoco::GlfwAdapter>());

    // sim->agent.SetTaskList(std::move(tasks));
    // std::string task_name = absl::GetFlag(FLAGS_task);

    // mjModel *m;
    // mjData *d;

    // m = mj_loadModel("../intvel.xml", nullptr);
    // d = mj_makeData(m);

    // sim->mnew = m;
    // sim->dnew = d;

    // sim->loadrequest = 2;

    // mjpc::ThreadPool physics_pool(1);
    // physics_pool.Schedule([]() { PhysicsLoop(*sim.get()); });

    sim->renderloop();

    sim.release();
    return 0;
}
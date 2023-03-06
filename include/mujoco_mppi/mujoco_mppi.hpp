#include <unistd.h>

#include <mujoco/mujoco.h>
#include <simulate.h>
#include <array_safety.h>

namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

class MujocoMPPI
{
public:
    MujocoMPPI(mjModel *model, mjData *data);

    void command();

private:
    mjModel *model_;
    mjData *data_;
};
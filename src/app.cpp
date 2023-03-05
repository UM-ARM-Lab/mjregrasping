#include <unistd.h>
#include <iostream>
#include <mujoco/mujoco.h>
#include <simulate.h>
#include <array_safety.h>
#include <glfw_adapter.h>

namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

// constants
const auto MUJOCO_PLUGIN_DIR = "mujoco_plugin";
const double syncMisalign = 0.1;       // maximum mis-alignment before re-sync (simulation seconds)
const double simRefreshFraction = 0.7; // fraction of refresh available for simulation
const int kErrorLength = 1024;         // load error string length

// model and data
mjModel *m = nullptr;
mjData *d = nullptr;

// control noise variables
mjtNum *ctrlnoise = nullptr;

using Seconds = std::chrono::duration<double>;

std::string getExecutableDir() {
  constexpr char kPathSep = '/';
  const char* path = "/proc/self/exe";
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new(std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();

  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}



// scan for libraries in the plugin directory to load additional plugins
void scanPluginLibraries() {
  // check and print plugins that are linked directly into the executable
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }

  const std::string sep = "/";

  // try to open the ${EXECDIR}/plugin directory
  // ${EXECDIR} is the directory containing the simulate binary itself
  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    return;
  }

  const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char* filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}


mjModel *LoadModel(const char *file, mj::Simulate &sim)
{
    // this copy is needed so that the mju::strlen call below compiles
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    // make sure filename is not empty
    if (!filename[0])
    {
        return nullptr;
    }

    // load and compile
    char loadError[kErrorLength] = "";
    mjModel *mnew = 0;
    if (mju::strlen_arr(filename) > 4 &&
        !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                      mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
        mnew = mj_loadModel(filename, nullptr);
        if (!mnew)
        {
            mju::strcpy_arr(loadError, "could not load binary model");
        }
    }
    else
    {
        mnew = mj_loadXML(filename, nullptr, loadError, mj::Simulate::kMaxFilenameLength);
        // remove trailing newline character from loadError
        if (loadError[0])
        {
            int error_length = mju::strlen_arr(loadError);
            if (loadError[error_length - 1] == '\n')
            {
                loadError[error_length - 1] = '\0';
            }
        }
    }

    mju::strcpy_arr(sim.loadError, loadError);

    if (!mnew)
    {
        std::printf("%s\n", loadError);
        return nullptr;
    }

    // compiler warning: print and pause
    if (loadError[0])
    {
        // mj_forward() below will print the warning message
        std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
        sim.run = 0;
    }

    sim.run = 0;

    return mnew;
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate &sim)
{
    // cpu-sim syncronization point
    std::chrono::time_point<mj::Simulate::Clock> syncCPU;
    mjtNum syncSim = 0;

    // run until asked to exit
    while (!sim.exitrequest.load())
    {
        if (sim.droploadrequest.load())
        {
            mjModel *mnew = LoadModel(sim.dropfilename, sim);
            sim.droploadrequest.store(false);

            mjData *dnew = nullptr;
            if (mnew)
            {
                dnew = mj_makeData(mnew);
            }
            if (dnew)
            {
                sim.load(sim.dropfilename, mnew, dnew);

                mj_deleteData(d);
                mj_deleteModel(m);

                m = mnew;
                d = dnew;
                mj_forward(m, d);

                // allocate ctrlnoise
                // it's zero
                free(ctrlnoise);
                ctrlnoise = (mjtNum *)malloc(sizeof(mjtNum) * m->nu);
                mju_zero(ctrlnoise, m->nu);
            }
        }

        if (sim.uiloadrequest.load())
        {
            sim.uiloadrequest.fetch_sub(1);
            mjModel *mnew = LoadModel(sim.filename, sim);
            mjData *dnew = nullptr;
            if (mnew)
                dnew = mj_makeData(mnew);
            if (dnew)
            {
                sim.load(sim.filename, mnew, dnew);

                mj_deleteData(d);
                mj_deleteModel(m);

                m = mnew;
                d = dnew;
                mj_forward(m, d);

                // allocate ctrlnoise
                free(ctrlnoise);
                ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
                mju_zero(ctrlnoise, m->nu);
            }
        }

        // sleep for 1 ms or yield, to let main thread run
        //  yield results in busy wait - which has better timing but kills battery life
        if (sim.run && sim.busywait)
        {
            std::this_thread::yield();
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        {
            // lock the sim mutex
            const std::lock_guard<std::mutex> lock(sim.mtx);

            // run only if model is present
            if (m)
            {
                // running
                if (sim.run)
                {
                    // record cpu time at start of iteration
                    const auto startCPU = mj::Simulate::Clock::now();

                    // elapsed CPU and simulation time since last sync
                    const auto elapsedCPU = startCPU - syncCPU;
                    double elapsedSim = d->time - syncSim;

                    // inject noise
                    if (sim.ctrlnoisestd)
                    {
                        // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
                        mjtNum rate = mju_exp(-m->opt.timestep / mju_max(sim.ctrlnoiserate, mjMINVAL));
                        mjtNum scale = sim.ctrlnoisestd * mju_sqrt(1 - rate * rate);

                        for (int i = 0; i < m->nu; i++)
                        {
                            // update noise
                            ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

                            // apply noise
                            d->ctrl[i] = ctrlnoise[i];
                        }
                    }

                    // requested slow-down factor
                    double slowdown = 100 / sim.percentRealTime[sim.realTimeIndex];

                    // misalignment condition: distance from target sim time is bigger than syncmisalign
                    bool misaligned =
                        mju_abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

                    // out-of-sync (for any reason): reset sync times, step
                    if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                        misaligned || sim.speedChanged)
                    {
                        // re-sync
                        syncCPU = startCPU;
                        syncSim = d->time;
                        sim.speedChanged = false;

                        // clear old perturbations, apply new
                        mju_zero(d->xfrc_applied, 6 * m->nbody);
                        sim.applyposepertubations(0); // move mocap bodies only
                        sim.applyforceperturbations();

                        // run single step, let next iteration deal with timing
                        mj_step(m, d);
                    }

                    // in-sync: step until ahead of cpu
                    else
                    {
                        bool measured = false;
                        mjtNum prevSim = d->time;

                        double refreshTime = simRefreshFraction / sim.refreshRate;

                        // step while sim lags behind cpu and within refreshTime
                        while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                               mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
                        {
                            // measure slowdown before first step
                            if (!measured && elapsedSim)
                            {
                                sim.measuredSlowdown =
                                    std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                                measured = true;
                            }

                            // clear old perturbations, apply new
                            mju_zero(d->xfrc_applied, 6 * m->nbody);
                            sim.applyposepertubations(0); // move mocap bodies only
                            sim.applyforceperturbations();

                            // call mj_step
                            mj_step(m, d);

                            // break if reset
                            if (d->time < prevSim)
                            {
                                break;
                            }
                        }
                    }
                }

                // paused
                else
                {
                    // apply pose perturbation
                    sim.applyposepertubations(1); // move mocap and dynamic bodies

                    // run mj_forward, to update rendering and joint sliders
                    mj_forward(m, d);
                }
            }
        } // release std::lock_guard<std::mutex>
    }
}

void PhysicsThread(mj::Simulate *sim, const char *filename)
{
    // request loadmodel if file given (otherwise drag-and-drop)
    if (filename != nullptr)
    {
        m = LoadModel(filename, *sim);
        if (m)
            d = mj_makeData(m);
        if (d)
        {
            sim->load(filename, m, d);
            mj_forward(m, d);

            // allocate ctrlnoise
            free(ctrlnoise);
            ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
            mju_zero(ctrlnoise, m->nu);
        }
    }

    PhysicsLoop(*sim);

    // delete everything we allocated
    free(ctrlnoise);
    mj_deleteData(d);
    mj_deleteModel(m);
}

// controller
extern "C"
{
    void controller(const mjModel *m, mjData *d);
}

// controller callback
void controller(const mjModel *m, mjData *data)
{
}

int main(int argc, char **argv)
{
    std::printf("MuJoCo Version: %s\n", mj_versionString());
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have Different versions");
    }

    scanPluginLibraries();

    auto sim = std::make_unique<mujoco::Simulate>(std::make_unique<mujoco::GlfwAdapter>());

    const char *filename = "../untangle.xml";

    mjcb_control = controller;

    std::thread physics_thread_handle(&PhysicsThread, sim.get(), filename);

    sim->renderloop();

    physics_thread_handle.join();

    return EXIT_SUCCESS;
}
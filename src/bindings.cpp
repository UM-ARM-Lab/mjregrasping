#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mujoco_mppi/mujoco_mppi.hpp>

namespace py = pybind11;

PYBIND11_MODULE(pymoveit_mppi, m)
{
    py::class_<MujocoMPPI>(m, "MujocoMPPI")
        .def(py::init<mjModel *, mjData *>(), py::arg("model"), py::arg("data"))
        //
        ;
}
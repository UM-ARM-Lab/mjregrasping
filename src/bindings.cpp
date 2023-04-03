#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mjregrasping/mjregrasping.h>

namespace py = pybind11;

PYBIND11_MODULE(pymjregrasping, m)
{
    m.doc() = "pymjregrasping module";

    m.def("preallocate_data_for_threads", &preallocate_data_for_threads);
    m.def("parallel_rollout", &parallel_rollout);
}
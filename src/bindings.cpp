#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mjregrasping/mjregrasping.h>

namespace py = pybind11;

PYBIND11_MODULE(pymjregrasping, m)
{
  m.doc() = "pymjregrasping module";
}
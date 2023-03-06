# MJRegrasping

Experimenting with using mujoco from C++.

TODO:

- [x] benchmark
- [x] fix joint control. The current joints are controlled with very weak position servos. Instead, we should control them with intvelocity actuators, then wrap that in a controller in C++ that takes position commands and sets the target for the invelocity controller.
- [ ] Write a C++ library that registers the custom callback and takes in the `mjmodel` and `mjdata`
- [ ] Demonstrate using this library from python, using the mujoco python bindings, but not `dm_control`

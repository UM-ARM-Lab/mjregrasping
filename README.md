# MJRegrasping

Experimenting with using mujoco from C++.

TODO:

- [ ] benchmark
- [ ] fix joint control. The current joints are controlled with very weak position servos. Instead, we should control them with intvelocity actuators, then wrap that in a controller in C++ that takes position commands and sets the target for the invelocity controller.

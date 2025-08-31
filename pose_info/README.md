This directory contains pose priors for YCB objects used during initialization.

- Files are named `<object_name>.npz` and typically contain wrist/world transforms from AnyGrasp or processed variants.
- These are consumed by `code/init_pos_multi.py` and `code/sim_gen.py` to initialize the gripper relative to objects.
- `width_info.csv` stores object width metadata used for heuristics.

Subdirectories:
- `init_opt/`: cached initial finger transforms per object/pose (see its README)

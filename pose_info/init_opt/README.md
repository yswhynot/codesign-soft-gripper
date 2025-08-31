Cached initial two-finger transforms for each object/pose.

Contents:
- `<object_name>.npz`: stores an array of two finger transforms per pose.
- `ood/`: out-of-distribution objects with the same format.

How they are created:
```bash
python code/init_pos_multi.py --object_name 006_mustard_bottle --num_frames 30 --train_iters 15000
```
At runtime, `code/sim_gen.py` will try to load these cached transforms to skip the expensive initialization step. If missing, it will compute and save them for future runs.

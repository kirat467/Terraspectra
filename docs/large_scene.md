# Large scene generator (Day 20)

`data/make_large_scene.py` writes a 1024 x 1024 x 200 simulated cube (about 15 km across at 15 m pixels, 419 MB as uint16) in strips of rows into a memory-mapped `.npy`, plus a label map. Nothing needs the whole cube in RAM.

    python data/make_large_scene.py --size 1024 --step 5
    cube = np.load("data/generated/large_cube_5.npy", mmap_mode="r")    # reads only the rows you touch

## Measured (this sandbox, 1024 x 1024, step 5)
| Rows per strip | Time | Peak resident memory |
|---|---|---|
| 8 | 11.8 s | 538 MB |
| 16 (default) | about 13 s | about 650 MB (interpolated, not measured) |
| 32 | 16.8 s | 859 MB |
| 128 | 19.9 s | 1959 MB |

Peak memory includes the memory-mapped file pages as they are written, so it is not only the strip. Smaller strips use less memory and cost a little time. These numbers are the starting point for Saturday's memory audit (batch size against peak RAM when training on patches from this file).

Class pixels at step 5: healthy 372,784; early stress 204,126; visible stress 95,594; not a crop 376,072.

## Honest limits
- Fields are a jittered grid, not a real farm layout, and outbreaks are random Gaussian blobs.
- The output does not depend on the strip size (one random seed per row), so runs are reproducible.
- Still simulated: it says how the pipeline scales, not how accurate the model is on real land.

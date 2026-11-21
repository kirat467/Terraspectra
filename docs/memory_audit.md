# Memory audit (Day 21)

`data/memory_audit.py` measures peak memory for reading and training on the 1024 x 1024 x 200 cube from `data/make_large_scene.py` (419 MB on disk as uint16). Each setting runs in a fresh Python process. Results: `docs/memory_audit.json` and the chart `docs/memory_audit.png`.

    python data/make_large_scene.py --size 1024 --step 5
    python data/memory_audit.py --cube data/generated/large_cube_5.npy

Two numbers are reported. **Resident** is what the operating system says the process holds, which includes pages of a memory-mapped file it has read in (those can be dropped when memory is short). **Heap** is what the program allocated itself with numpy (cannot be dropped; torch's internal buffers are not counted, so for training read the resident figure).

## Results (this sandbox, CPU only; Python and torch with no data already use about 510 MB)
| Experiment | Setting | Resident MB | Heap MB |
|---|---|---|---|
| Read patches, load whole cube | batch 16 to 1024 | 638 | 630 |
| Read patches, memory-mapped | batch 16 | 353 | 211 |
| Read patches, memory-mapped | batch 256 | 518 | 211 |
| Read patches, memory-mapped | batch 4096 | 1,202 | 558 |
| PCA of whole scene in strips | 8 to 32 rows | 620 | 211 |
| PCA of whole scene in strips | 128 rows | 910 | 399 |
| PCA of whole scene in strips | 256 rows | 1,200 | 714 |
| 3D-CNN step (9 x 9 x 20 patches) | batch 16 | 604 | n/a |
| 3D-CNN step | batch 256 | 669 | n/a |
| 3D-CNN step | batch 1024 | 811 | n/a |
| 3D-CNN step | batch 4096 | 1,495 | n/a |

## What it says
- **Memory-mapping saves about 420 MB of heap** (the cube itself) and loses nothing in speed for random patch reads.
- **Strips of 32 rows or fewer cost nothing extra**; 128 rows is about 300 MB more and 256 about 600 MB more. Use 32.
- **Batch size is the main training cost above about 1,000 patches**: roughly 200 MB per 1,000 patches (about 0.2 MB per patch; the input alone is only 6.5 KB, so almost all of it is the network's activations and gradients). Use 256 on a laptop.
- The 210 MB fixed heap in the first two experiments is the PCA fit (64 whole rows, covariance in float64), not the batch.
- A 4 GB machine is fine with memory-mapping, strip 32, batch 256 (under 700 MB resident including torch). Loading the whole cube and a batch of 4096 would need over 1.2 GB.

## Limits
Resident figures include file pages the system chose to read ahead, so they move a little between runs (one earlier run showed 688 MB for a setting that normally gives 638). Training was timed and measured for one step, not a full epoch, and only on CPU. A real 1024 x 1024 scene with more bands would be proportionally larger.

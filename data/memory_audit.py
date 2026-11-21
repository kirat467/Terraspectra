"""Memory audit: how much RAM does it take to read a big scene and train on it, and how does that change with batch and strip size?

Every measurement runs in its own fresh Python process, so the peak is clean and not polluted by earlier runs. Two numbers per run:
peak_mb = resident memory as the operating system reports it (includes pages of a memory-mapped file that the system has read
in; those can be dropped under pressure) and heap_mb = what the program allocated itself with numpy (the part that cannot be
dropped; torch's internal buffers are not in it, so read the train experiment from peak_mb). Three experiments on a large cube such as data/generated/large_cube_5.npy
(make it with data/make_large_scene.py):

  load      : how the cube is read. "full" loads it all into RAM; "mmap" opens it memory-mapped and touches only what is needed.
  project   : reduce 200 bands to 20 with PCA over the whole scene, in strips of rows. Strip size against peak memory.
  train     : one forward and backward step of the 3D-CNN on 9 x 9 x 20 patches, batch size against peak memory.

    python data/memory_audit.py --cube data/generated/large_cube_5.npy --out docs/memory_audit.json
    python data/memory_audit.py --cube ... --quick        # fewer settings, for a fast check

Output: a JSON file (every setting with its peak MB and seconds) and a PNG chart next to it.
"""
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCALE = 10000.0
PATCH, COMPONENTS = 9, 20


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def fit_pca(cube, n_components=COMPONENTS, n_rows=64, seed=0):
    """Mean and top principal axes from `n_rows` random whole rows (contiguous reads, so a memory-mapped cube stays mostly unread)."""
    rng = np.random.default_rng(seed)
    h, w, b = cube.shape
    rows = np.sort(rng.choice(h, size=min(n_rows, h), replace=False))
    x = np.concatenate([np.asarray(cube[r], dtype=np.float32) for r in rows]) / SCALE
    mean = x.mean(0)
    _, vecs = np.linalg.eigh(np.cov((x - mean).T))
    return mean, vecs[:, ::-1][:, :n_components].astype(np.float32)


def project_in_strips(cube, mean, axes, strip):
    """(h, w, 200) uint16 -> (h, w, 20) float32, `strip` rows at a time."""
    h, w, _ = cube.shape
    out = np.empty((h, w, axes.shape[1]), dtype=np.float32)
    for r in range(0, h, strip):
        block = np.asarray(cube[r:r + strip], dtype=np.float32) / SCALE
        out[r:r + strip] = (block - mean) @ axes
    return out


def worker(args):
    import tracemalloc
    tracemalloc.start()
    t0 = time.time()
    path = args.cube
    if args.exp == "load":
        cube = np.load(path) if args.setting == "full" else np.load(path, mmap_mode="r")
        mean, axes = fit_pca(cube)
        rng = np.random.default_rng(1)
        n = args.batch
        ys, xs = rng.integers(0, cube.shape[0] - PATCH, n), rng.integers(0, cube.shape[1] - PATCH, n)
        batch = np.stack([np.asarray(cube[y:y + PATCH, x:x + PATCH], dtype=np.float32) for y, x in zip(ys, xs)]) / SCALE
        _ = (batch - mean) @ axes                                    # the batch a training step would receive
    elif args.exp == "project":
        cube = np.load(path, mmap_mode="r")
        mean, axes = fit_pca(cube)
        _ = project_in_strips(cube, mean, axes, args.strip)
    else:                                                          # train
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml" / "src"))
        from model_3dcnn import Crop3DCNN
        torch.manual_seed(0)
        model = Crop3DCNN(3)
        x = torch.randn(args.batch, 1, COMPONENTS, PATCH, PATCH)
        loss = torch.nn.functional.cross_entropy(model(x), torch.zeros(args.batch, dtype=torch.long))
        loss.backward()
    heap = tracemalloc.get_traced_memory()[1] / 1e6          # arrays the program itself allocated (not memory-mapped pages, not torch internals)
    print(json.dumps({"peak_mb": round(peak_mb(), 1), "heap_mb": round(heap, 1), "seconds": round(time.time() - t0, 2)}))


def run(exp, setting, cube, batch=0, strip=0):
    cmd = [sys.executable, "-I", __file__, "--worker", "--exp", exp, "--setting", setting, "--cube", str(cube), "--batch", str(batch), "--strip", str(strip)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    return json.loads(out)


def baseline_mb():
    """Peak memory of an empty Python with numpy (and torch for the train experiment) imported, to subtract from nothing and show alongside."""
    code = "import resource,numpy;print(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))"
    code_t = "import resource,numpy,torch;print(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))"
    f = lambda c: float(subprocess.run([sys.executable, "-I", "-c", c], capture_output=True, text=True).stdout.strip() or 0)
    return {"python_numpy_mb": f(code), "python_numpy_torch_mb": f(code_t)}


def audit(cube, quick=False):
    cube = Path(cube)
    arr = np.load(cube, mmap_mode="r")
    info = {"cube": cube.name, "shape": list(arr.shape), "cube_file_mb": round(cube.stat().st_size / 1e6, 1), "baseline": baseline_mb()}
    batches = [16, 256, 4096] if quick else [16, 64, 256, 1024, 4096]
    strips = [16, 128] if quick else [8, 16, 32, 64, 128, 256]
    info["load"] = [{"setting": s, "batch": b, **run("load", s, cube, batch=b)} for s in ("full", "mmap") for b in batches]
    info["project"] = [{"strip": s, **run("project", "mmap", cube, strip=s)} for s in strips]
    info["train"] = [{"batch": b, **run("train", "cpu", cube, batch=b)} for b in batches]
    return info


def chart(info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), dpi=140)
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    for setting, colour in (("full", orange), ("mmap", blue)):
        rows = [r for r in info["load"] if r["setting"] == setting]
        ax[0].plot([r["batch"] for r in rows], [r["peak_mb"] for r in rows], "o-", color=colour, lw=2)
        ax[0].plot([r["batch"] for r in rows], [r["heap_mb"] for r in rows], "o--", color=colour, lw=1, alpha=.55, ms=3)
        ax[0].annotate("load all" if setting == "full" else "memory-map", (rows[0]["batch"], rows[0]["peak_mb"]), xytext=(4, 8 if setting == "full" else -14), textcoords="offset points", ha="left", color=colour, fontsize=9)
    ax[0].set_xscale("log"); ax[0].set_title("Reading patches", fontsize=10); ax[0].set_xlabel("patches per batch"); ax[0].set_ylabel("peak memory, MB (solid: resident, dashed: heap)")
    ax[1].plot([r["strip"] for r in info["project"]], [r["peak_mb"] for r in info["project"]], "o-", color=aqua, lw=2)
    ax[1].plot([r["strip"] for r in info["project"]], [r["heap_mb"] for r in info["project"]], "o--", color=aqua, lw=1, alpha=.55, ms=3)
    ax[1].set_xscale("log", base=2); ax[1].set_title("PCA over the whole scene, in strips", fontsize=10); ax[1].set_xlabel("rows per strip")
    ax[2].plot([r["batch"] for r in info["train"]], [r["peak_mb"] for r in info["train"]], "o-", color=blue, lw=2)
    ax[2].axhline(info["baseline"]["python_numpy_torch_mb"], color="#8a948d", ls="--", lw=1)
    ax[2].annotate("Python + torch, no data", (info["train"][0]["batch"], info["baseline"]["python_numpy_torch_mb"]), xytext=(4, 5), textcoords="offset points", fontsize=8, color="#5b665f")
    ax[2].set_xscale("log"); ax[2].set_title("3D-CNN step (CPU)", fontsize=10); ax[2].set_xlabel("batch size")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.25); a.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cube", default="data/generated/large_cube_5.npy")
    ap.add_argument("--out", default="docs/memory_audit.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--exp"); ap.add_argument("--setting", default="mmap")
    ap.add_argument("--batch", type=int, default=0); ap.add_argument("--strip", type=int, default=0)
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    info = audit(a.cube, a.quick)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    chart(info, out.with_suffix(".png"))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

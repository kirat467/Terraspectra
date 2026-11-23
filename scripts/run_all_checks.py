"""Run every check in the repository and print one summary: Python tests, the frontend zone tests, and the frontend build.

    python scripts/run_all_checks.py            # everything
    python scripts/run_all_checks.py --fast     # skip the slow tests (memory audit, large scene) and the frontend build

Exit code 0 means everything passed. Run from anywhere; paths are relative to the repository root.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLOW = ["data/tests/test_memory_audit.py", "data/tests/test_make_large_scene.py"]


def run(name, cmd, cwd=ROOT):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    lines = p.stdout.strip().splitlines() or p.stderr.strip().splitlines() or [""]
    tail = next((l.lstrip("# ") for l in lines if l.startswith("# pass")), lines[-1])        # node's test runner prints "# pass N"
    return {"name": name, "ok": p.returncode == 0, "seconds": round(time.time() - t0, 1), "note": tail[:90], "output": p.stdout + p.stderr}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    ignore = [f"--ignore={s}" for s in SLOW] if a.fast else []
    results = [run("data tests", [sys.executable, "-m", "pytest", "data/tests", "-q", "-p", "no:cacheprovider", *ignore]),
               run("ml tests", [sys.executable, "-m", "pytest", "ml/src", "-q", "-p", "no:cacheprovider"])]
    node = shutil.which("node")
    results.append(run("frontend zone tests", [node, "frontend/scripts/test_zones.mjs"]) if node else {"name": "frontend zone tests", "ok": None, "seconds": 0, "note": "node not installed, skipped", "output": ""})
    npm = shutil.which("npm")
    if a.fast:
        results.append({"name": "frontend build", "ok": None, "seconds": 0, "note": "skipped (--fast)", "output": ""})
    elif npm and (ROOT / "frontend" / "node_modules").exists():
        results.append(run("frontend build", [npm, "run", "build", "--", "--outDir", str(ROOT / "frontend" / ".check_build")], cwd=ROOT / "frontend"))
        shutil.rmtree(ROOT / "frontend" / ".check_build", ignore_errors=True)
    else:
        results.append({"name": "frontend build", "ok": None, "seconds": 0, "note": "run `npm install` in frontend/ first, skipped", "output": ""})
    print()
    for r in results:
        mark = {True: "PASS", False: "FAIL", None: "skip"}[r["ok"]]
        print(f"{mark}  {r['name']:22s} {r['seconds']:6.1f}s  {r['note']}")
    failed = [r for r in results if r["ok"] is False]
    for r in failed:
        print(f"\n--- {r['name']} output (last 30 lines) ---\n" + "\n".join(r["output"].splitlines()[-30:]))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

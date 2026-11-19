import fcntl
import time
from pathlib import Path
target=Path("results/Hopper-v3/EQL/seed204")
target.mkdir(exist_ok=True)
with (target/"run.lock").open("a") as lock:
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print("Already being trained",flush=True)
        raise SystemExit(0)
    if (target/"progress.json").exists():
        raise SystemExit(0)
    print("Reserved unstarted Hopper-v3 EQL seed 204 for L40",flush=True)
    until=time.monotonic()+20*60
    while time.monotonic()<until and not Path("results/Walker2d-v3/SAC100/seed204/final.json").exists():
        time.sleep(2)
    print("Released for L40 pickup",flush=True)

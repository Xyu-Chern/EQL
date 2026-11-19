import os
import socket
import subprocess
import sys
import time
from pathlib import Path

root=Path("/gscratch/stf/xyuchern/eql-million-hopper-walker-v3")
node=socket.gethostname().split(".")[0]
task,method=sys.argv[1:3]
code=Path(__file__).resolve().parent

def done(task,method):
    return all((root/"results"/task/method/f"seed{s}"/"final.json").is_file() for s in (201,202,203,204))

def command(task,method):
    workdir=code/("adaptive" if method=="Adaptive11" else "fixed")
    return workdir, [sys.executable,"-u",str(root/"posix-lock-runner.py"),str(workdir/"run_million_native.py"),
                     task,method,*sys.argv[3:]]

if node=="g3097" and (task,method)==("Hopper-v3","Adaptive11") and not done("Hopper-v3","EQL"):
    # Let the current 2080 Ti process finish or release its last checkpoint.
    while not done("Hopper-v3","EQL"):
        q=subprocess.run(["squeue","-h","-j","43357300","-o","%T"],capture_output=True,text=True)
        if q.stdout.strip() not in ("RUNNING","COMPLETING"): break
        time.sleep(3)
    if not done("Hopper-v3","EQL"):
        wd,cmd=command("Hopper-v3","EQL")
        subprocess.run(cmd,cwd=wd,check=True)

# These four groups began with node-local flock. The designated original worker
# finishes each group; the L40 also handles the 2080 Ti handoff above.
owners={("Walker2d-v3","EQL_bagging"):"g3047",("Walker2d-v3","Adaptive11"):"g3034",
        ("Walker2d-v3","SAC100"):"g3097",("Hopper-v3","EQL"):"g3097"}
if done(task,method) or node!=owners.get((task,method),node):
    sys.exit(0)
workdir,cmd=command(task,method)
os.chdir(workdir)
os.execv(sys.executable,cmd)

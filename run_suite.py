"""Run the two online tasks sequentially, retaining logs and GPU telemetry."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
root=Path('results/online_v1')
root.mkdir(parents=True,exist_ok=True)
stop=threading.Event()
def monitor():
    with (root/'gpu_telemetry.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['unix_time','memory_used_MiB_total_device','utilization_percent','temperature_C','power_W'])
        while not stop.is_set():
            p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],capture_output=True,text=True)
            if p.returncode==0:
                w.writerow([time.time(),*[x.strip() for x in p.stdout.strip().split(',')]])
                f.flush()
            stop.wait(5)
t=threading.Thread(target=monitor,daemon=True)
t.start()
try:
    for env in ['Pendulum-v1','Reacher-misc']:
        for group in ['A','B','C','D']:
            # Isolate XLA contexts across static algorithm configurations.
            # Multi-group process stalled in initialization after donating buffers.
            command=[sys.executable,'-u','run_online.py','--env',env,'--groups',group]
            with (root/f'{env}_{group}.log').open('w') as log:
                p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in p.stdout:
                    log.write(line)
                    log.flush()
                    print(line,end='',flush=True)
                code=p.wait()
                if code:
                    raise RuntimeError(f'{env} {group} failed with exit code {code}')
finally:
    stop.set()
    t.join(timeout=10)

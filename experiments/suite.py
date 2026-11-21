"""Run the online or original baseline suite in isolated JAX processes."""
import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def monitor_gpu(output, stop):
    with (output / "gpu_telemetry.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["unix_time", "memory_used_MiB_total_device", "utilization_percent", "temperature_C", "power_W"])
        while not stop.is_set():
            sample = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
            )
            if sample.returncode == 0:
                writer.writerow([time.time(), *(value.strip() for value in sample.stdout.strip().split(","))])
                stream.flush()
            stop.wait(5)


def run(suite, output):
    if suite == "online":
        module, option, groups = "experiments.online", "--groups", ("A", "B", "C", "D")
    else:
        module, option, groups = "experiments.baseline", "--method", ("SAC", "EQL", "EQL_bagging")
    output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    telemetry = threading.Thread(target=monitor_gpu, args=(output, stop), daemon=True)
    telemetry.start()
    try:
        for task in ("Pendulum-v1", "Reacher-misc"):
            for group in groups:
                command = [sys.executable, "-u", "-m", module, "--env", task, option, group, "--output", str(output)]
                with (output / f"{task}_{group}.log").open("w") as log, subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                    env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
                ) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    if process.wait():
                        raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        stop.set()
        telemetry.join(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("online", "baseline"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default = "online_v1" if args.suite == "online" else "original_comparison"
    run(args.suite, args.output or Path("results") / default)

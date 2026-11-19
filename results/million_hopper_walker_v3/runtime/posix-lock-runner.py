"""Use cluster-wide POSIX record locks on GPFS for the native experiment."""
import fcntl
import runpy
import sys
from pathlib import Path

file=Path(sys.argv[1]).resolve()
sys.argv=sys.argv[1:]
sys.path.insert(0,str(file.parent))
fcntl.flock=fcntl.lockf
runpy.run_path(str(file),run_name="__main__")

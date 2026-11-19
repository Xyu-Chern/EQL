import fcntl,sys,time
from pathlib import Path
files=[]
for method in ['flock','lockf']:
    f=open('test-lock-'+method,'a');files.append(f)
    try:getattr(fcntl,method)(f,fcntl.LOCK_EX|fcntl.LOCK_NB);print(method,'acquired',flush=True)
    except BlockingIOError:print(method,'blocked',flush=True)
if sys.argv[1]=='hold':
    Path('test-lock-ready').touch()
    time.sleep(12)

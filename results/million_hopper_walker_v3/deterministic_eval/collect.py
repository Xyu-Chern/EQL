import os,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'deterministic_eval'/'checkpoint_links'
while True:
    for origin in [root/'results',root/'backups'/'rtx2080ti'/'20261118-174106'/'results']:
        for p in origin.glob('*/*/seed*/resume.msgpack'):
            d=out/p.parent.relative_to(origin)
            d.mkdir(parents=True,exist_ok=True)
            try:
                s=p.stat()
                target=d/f'{s.st_mtime_ns}-{s.st_ino}.msgpack'
                if not target.exists():os.link(p,target)
            except FileNotFoundError:pass
    if sum(1 for _ in (root/'results').glob('*/*/seed*/final.json'))==40:break
    time.sleep(4)

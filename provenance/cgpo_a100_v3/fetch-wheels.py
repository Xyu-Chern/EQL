import concurrent.futures,hashlib,json,sys,urllib.request
from pathlib import Path
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
tags={t:i for i,t in enumerate(sys_tags())}
dest=Path(sys.argv[2])
def download(req):
    name,ver=req.split('==')
    data=json.load(urllib.request.urlopen('https://pypi.org/pypi/'+name+'/'+ver+'/json',timeout=30))
    candidates=[]
    for f in data['urls']:
        if f['filename'].endswith('.whl'):
            compatible=set(parse_wheel_filename(f['filename'])[3]) & tags.keys()
            if compatible:candidates.append((min(tags[t] for t in compatible),f['filename'],f))
    _,_,f=min(candidates)
    path=dest/f['filename']
    if path.exists() and path.stat().st_size==f['size']:return
    tmp=path.with_suffix('.part')
    with urllib.request.urlopen(f['url'],timeout=90) as src,tmp.open('wb') as out:
        h=hashlib.sha256()
        while b:=src.read(1024*1024):
            out.write(b);h.update(b)
    assert h.hexdigest()==f['digests']['sha256']
    tmp.replace(path);print(name,ver,round(path.stat().st_size/2**20),'MiB',flush=True)
reqs=[line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip() and not line.startswith('#')]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(download,reqs))

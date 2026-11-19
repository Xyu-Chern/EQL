import hashlib, json
from pathlib import Path
import jax,numpy as np
from flax import serialization as s

root=Path(__file__).resolve().parents[1]
base=root/'deterministic_eval'
keys=set()
for kind,source in [('resume','checkpoint_links'),('actor','extra_actors')]:
    for p in sorted((base/source).glob('*/*/seed*/*.msgpack')):
        obj=s.msgpack_restore(p.read_bytes())
        if kind=='resume':
            train=obj['training']
            step=int(train['global_step']); actor=train['actor_ts']['params']
        else:
            step=int(p.stem);actor=obj
        task,method,seed=p.parts[-4:-1]
        score=base/'scores'/task/method/seed/'points'/f'{step:07d}.json'
        assert score.is_file(),str(score)
        digest=hashlib.sha256()
        for a in jax.tree.leaves(actor):digest.update(np.asarray(a).tobytes())
        assert digest.hexdigest()==json.loads(score.read_bytes())['actor_sha256']
        target=base/'actors'/task/method/seed/f'{step:07d}.msgpack'
        if target not in keys:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(s.msgpack_serialize(actor));keys.add(target)
points=list((base/'scores').glob('*/*/seed*/points/*.json'))
assert len(keys)+40==len(points)
print(len(keys),'intermediate actors archived;',len(points),'total points')

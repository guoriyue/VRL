"""Compare the first collection against its control before any parameter update."""
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
CONTROL = Path('/mnt/nvme/layering_annotation_pair_20261030/reviewed/rollouts')

def pixels(path):
    with Image.open(path) as image:
        return image.convert('RGBA').tobytes()

def indexed(root):
    result = {}
    for folder in sorted(root.iterdir())[:16]:
        key = (hashlib.sha256(pixels(folder / 'source.png')).hexdigest(), int(folder.name.rsplit(':', 1)[1]))
        assert key not in result
        result[key] = folder
    assert len(result) == 16
    return result

old, new = indexed(CONTROL), indexed(ROOT / 'rollouts')
assert old.keys() == new.keys()
records = []
for key, folder in new.items():
    control = old[key]
    names = sorted(p.name for p in folder.glob('layer_*.png'))
    assert names == sorted(p.name for p in control.glob('layer_*.png'))
    exact = all(pixels(folder / name) == pixels(control / name) for name in names)
    before = json.loads((control / 'scores.json').read_text())
    after = json.loads((folder / 'scores.json').read_text())
    unchanged = all(before[k] == after[k] for k in before if k != 'layered_grouping')
    residual = after['layered_grouping'] - (before['layered_grouping'] - after['excess_foreground_opacity'])
    records.append(dict(source=key[0],sample=key[1],saved_layers_exact=exact,old_terms_exact=unchanged,reward_formula_residual=residual,excess=after['excess_foreground_opacity'],supervised=after['integrity_supervised']))
passed = all(r['saved_layers_exact'] and r['old_terms_exact'] and abs(r['reward_formula_residual']) < 1e-10 for r in records)
report = dict(passed=passed,scope='First 16 saved PNG stacks, matched by source pixels and sample index; equality does not establish latent equality.',records=records)
(ROOT / 'initial_control_alignment.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=passed,candidates=len(records),nonzero_penalties=sum(r['excess']>0 for r in records))))
assert passed

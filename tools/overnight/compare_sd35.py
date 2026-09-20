import json,torch
from pathlib import Path
out=Path('outputs/overnight_20260918')
a=torch.load(out/'sd35/checkpoint-1/checkpoint.pt',map_location='cpu',weights_only=False)
b=torch.load(out/'sd35_parent/checkpoint-final/checkpoint.pt',map_location='cpu',weights_only=False)
x=a['trainer']['optimizer']['state'];y=b['trainer']['optimizer']['state']
print('keys',list(x)[:2]);diff=0.;norm=0.;dot=0.;other=0.;maxabs=0.
for k in x:
 if 'exp_avg' not in x[k]: continue
 l=x[k]['exp_avg'].double();r=y[k]['exp_avg'].double();d=l-r
 diff+=float(d.square().sum());norm+=float(l.square().sum());other+=float(r.square().sum());dot+=float((l*r).sum());maxabs=max(maxabs,float(d.abs().max()))
report={'first_moment_relative_l2':(diff/norm)**.5,'first_moment_cosine':dot/(norm*other)**.5,'first_moment_max_abs':maxabs}
(out/'sd35_parent/optimizer_comparison.json').write_text(json.dumps(report,indent=2));print(report)
print('RNG keys',len(a['rng']['by_rank']))

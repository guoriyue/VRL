"""Compare first-update Adam moments as a paired gradient-equivalence gate."""
import argparse,json,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
x=torch.load(a.reference/'checkpoint-1/checkpoint.pt',map_location='cpu',weights_only=False)['trainer']['optimizer']['state'];y=torch.load(a.candidate/'checkpoint-1/checkpoint.pt',map_location='cpu',weights_only=False)['trainer']['optimizer']['state'];assert x.keys()==y.keys()
diff=norm=0.;n=0
for k in x:
 if 'exp_avg' not in x[k]: continue
 l=x[k]['exp_avg'].double();r=y[k]['exp_avg'].double();diff+=float((l-r).square().sum());norm+=float(l.square().sum());n+=1
ratio=(diff/max(norm,1e-30))**.5
report={'gradient_relative_l2':ratio,'limit':1e-4,'tensors':n,'status':'passed' if n and ratio<=1e-4 else 'failed'}
(a.output/'rank0.json').write_text(json.dumps(report,indent=2));print(report,flush=True);assert report['status']=='passed'

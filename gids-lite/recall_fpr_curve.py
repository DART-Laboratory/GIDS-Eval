import os
import sys, csv
from pathlib import Path

import numpy as np
ROOT = os.environ.get("GIDS_EVAL_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, ROOT+'/EG7'); sys.path.insert(0, ROOT+'/src')
import eg7_worker as W
from sklearn.preprocessing import StandardScaler
CFG=sys.argv[1]; OUT=sys.argv[2]
CAP={}
_o=W.run_detection
def cap(*, config, bundle, **kw):
    CAP['c']=config; CAP['b']=bundle; return _o(config=config, bundle=bundle, **kw)
W.run_detection=cap
sys.argv=['x','--config',CFG]
try: W.main()
except SystemExit: pass
b=CAP['b']; cfg=CAP['c']
xtr,_=W.sanitize_feature_matrix(b.x_train); xte,_=W.sanitize_feature_matrix(b.x_test)
sc=StandardScaler().fit(xtr); utr=sc.transform(xtr); ute=sc.transform(xte)
yte=np.asarray(b.y_test).astype(int)
if cfg['loss']=='contrastive':
    m=W.train_contrastive_head(utr,int(b.rep_dim)); ts=W.score_linear_head(m,ute)
else:
    m=W.train_reconstruction_head(utr); ts=W.score_reconstruction(m,ute)
atk=ts[yte==1]; ben=ts[yte==0]
rows=[]
for tf in np.concatenate([np.geomspace(1e-4,0.5,120),[0.6,0.7,0.8,0.9,1.0]]):
    thr=np.quantile(ben,1.0-tf)
    rec=float((atk>=thr).mean()); fp=int((ben>=thr).sum())
    rows.append((cfg['dataset'], float(tf), fp/len(ben), rec))
with open(OUT,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['dataset','target_fpr','fpr','recall']); w.writerows(rows)
print('wrote',OUT,len(rows))

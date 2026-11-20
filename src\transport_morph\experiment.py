import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def dataset(n=900,seed=27):
 r=np.random.default_rng(seed);y=r.binomial(1,.5,n);samples=[]
 for label in y:
  mix=r.binomial(1,.35+label*.15,120);samples.append(r.normal(-.5+label*.12,.7,120)*(1-mix)+r.normal(1.2-label*.15,.45,120)*mix)
 return np.array(samples),y
def evaluate(seed=27):
 x,y=dataset(seed=seed);cut=600;q=np.linspace(.05,.95,25);transport=np.quantile(x,q,axis=1).T;summary=np.c_[x.mean(1),x.std(1),np.median(x,1)];out={}
 for name,f in {"summary_features":summary,"transport_embedding":transport}.items():
  m=LogisticRegression(max_iter=500).fit(f[:cut],y[:cut]);p=m.predict_proba(f[cut:])[:,1];out[name]={"auroc":float(roc_auc_score(y[cut:],p))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()

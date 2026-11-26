"""Three latent difficulty classes; learn a mixture of sequential observation models."""
from itertools import product
import numpy as np
import pandas as pd
from scipy.special import softmax
from search import BANK, CACHE, OUT, TIMES

def observations(path):
    bank=BANK/(path.stem+'.parquet')
    df=pd.read_parquet(bank,columns=['question_id','sample_index','prediction','answer_mean_logprob','answer_token_count','correct']).sort_values(['question_id','sample_index'])
    answers=df.prediction.fillna('').to_numpy().reshape(-1,16)
    q=len(answers)
    events=np.empty((q,16),int)
    for i,row in enumerate(answers):
        count={}
        for t,a in enumerate(row):
            events[i,t]=0 if not a else 1 if a not in count else 2 if count[a]==max(count.values()) else 3
            if a: count[a]=count.get(a,0)+1
    c=df.answer_mean_logprob.where(df.answer_token_count>0).to_numpy().reshape(q,16)
    return events,c,df.correct.to_numpy(int).reshape(q,16),df.question_id.to_numpy()[::16]

# First sample, second, 3-4, 5-8, and 9-16 have different repeat frequencies.
stage=np.array([0,1,2,2,3,3,3,3]+[4]*8)
def fit_filter(event,conf,y,train,with_conf,example_index):
    finite=np.isfinite(conf)
    seen=conf[train][finite[train]]
    cuts=np.quantile(seen,[1/3,2/3]) if seen.size else [0,0]
    c=np.where(finite,1+np.digitize(conf,cuts),0) if with_conf else np.zeros_like(event)
    bins=16 if with_conf else 4
    obs=event*(4 if with_conf else 1)+c
    codes=2*obs+y
    # Initial responsibilities use training accuracy only; EM then fits soft class assignments.
    freq=y[train].mean(1)
    resp=softmax(-20*(freq[:,None]-np.array([.08,.50,.92]))**2,axis=1)
    train_codes=codes[train]
    for iteration in range(500):
        prior=(resp.sum(0)+1)/(len(resp)+3)
        joint=np.empty((3,5,bins*2))
        for k in range(5):
            ix=stage==k
            for h in range(3):
                flat=train_codes[:,ix].ravel()
                counts=np.bincount(flat,weights=np.repeat(resp[:,h],sum(ix)),minlength=bins*2)
                joint[h,k]=(counts+.1)/(counts.sum()+.1*len(counts))
        logp=np.log(joint[:,stage[None,:],train_codes]).sum(-1).T
        updated=softmax(logp+np.log(prior),axis=1)
        if np.max(np.abs(updated-resp))<1e-5:break
        resp=updated
    else: raise RuntimeError('Mixture failed to converge')
    joint=joint.reshape(3,5,bins,2)
    model_accuracy=joint[:,:,:,1].sum(-1)[:,stage]
    order=np.argsort(model_accuracy.mean(1))
    model_accuracy=model_accuracy[order]
    emission=joint.sum(-1)[order]
    posterior=np.broadcast_to(prior[order],(len(y),3)).copy()
    p=[]; traced=[posterior[example_index].copy()]
    for j,k in enumerate(stage):
        posterior*=emission[:,k,obs[:,j]].T
        posterior/=posterior.sum(1,keepdims=True)
        traced.append(posterior[example_index].copy())
        if (j+1)%2==0:
            mean=model_accuracy[:,j+1:].mean(1) if j<15 else model_accuracy[:,-1]
            p.append((posterior*mean).sum(1))
    return np.stack(p,axis=1),traced,c[example_index]

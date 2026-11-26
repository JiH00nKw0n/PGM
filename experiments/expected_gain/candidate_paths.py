"""Use calibrated marginal correctness scores for the actual submitted candidates."""
from itertools import product
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from search import BANK,CACHE,OUT,TIMES,top_four

def prepare(path):
    bank=BANK/(path.stem+'.parquet')
    d=pd.read_parquet(bank,columns=['split','question_id','prediction','sample_index','correct','answer_mean_logprob','answer_token_count']).sort_values(['question_id','sample_index'])
    a=d.prediction.fillna('').to_numpy().reshape(-1,16)
    q=len(a); ix=np.arange(q)
    split=d.split.to_numpy()[::16]
    c=d.answer_mean_logprob.where(d.answer_token_count>0).to_numpy().reshape(q,16)
    mem=np.array([pd.factorize(r,sort=False)[0] for r in a])[:,:,None]==np.arange(16)
    label=np.take_along_axis(d.correct.to_numpy(bool).reshape(q,16),mem.argmax(1),axis=1)
    fs=[]
    for t in [1,*TIMES]:
        m=mem[:,:t]; v=m.sum(1); finite=np.isfinite(c[:,:t]); den=(m*finite[:,:,None]).sum(1)
        mean=(m*np.where(finite,c[:,:t],0)[:,:,None]).sum(1)/np.where(den,den,np.nan)
        top=(m*np.array([top_four(r) for r in c[:,:t]])[:,:,None]).sum(1)
        fs.append(np.stack([v,mean,top],axis=-1))
    full=fs[-1];active=full[:,:,0]>0; use=active&(split=='train')[:,None]
    fill=np.nan_to_num(np.nanmedian(full[use],axis=0));fs=[np.where(np.isfinite(f),f,fill) for f in fs]
    scale=StandardScaler().fit(fs[-1][use])
    w=np.broadcast_to(1/active.sum(1)[:,None],use.shape)[use]
    def scores(f,coef):return np.einsum('qki,i->qk',(f-scale.mean_)/scale.scale_,coef)
    best=-1
    for k in [.01,.1,1,10]:
        lr=LogisticRegression(C=k,solver='liblinear',max_iter=1000).fit(scale.transform(fs[-1][use]),label[use],sample_weight=w)
        coef=lr.coef_[0];lead=np.where(active,scores(fs[-1],coef),-np.inf).argmax(1)
        acc=label[ix,lead][split=='dev'].sum()
        if acc>best:best=acc;chosen=coef.copy()
    with np.load(path,allow_pickle=True) as z:
        s={k:z[k] for k in ['margin','valid','hit','tok','ids','split']}
    assert np.array_equal(s['ids'],d.question_id.to_numpy()[::16])
    nonempty = np.take_along_axis(a, mem.argmax(1), axis=1) != ""
    history=[];prev_lead=None;states=[]
    for j,f in enumerate(fs):
        raw=scores(f,chosen);seen=f[:,:,0]>0
        pair=np.argsort(-np.where(seen,raw,-np.inf),axis=1,kind='stable')[:,:2]
        leader,runner=pair.T
        use=seen&(split=='calibration')[:,None]
        cal=LogisticRegression(C=1,solver='liblinear').fit(np.r_[raw[use],0.,0.][:,None],np.r_[label[use],0,1])
        # Preserve raw candidate order. A nonpositive calibration slope gets a constant rate.
        prob=np.where(seen,expit(cal.intercept_[0]+cal.coef_[0,0]*raw),0)
        if cal.coef_[0,0]<=0:prob=np.where(seen,label[use].mean(),0.)
        if j:
            assert np.array_equal(label[ix,leader],s['hit'][:,j-1])
            p=prob[ix,leader]; backup=prob[ix,runner]
            same=leader==prev_lead
            # Compare the very same candidates at successive checks, even if their rank changed.
            drift=np.maximum(p-history[-1][ix,leader],backup-history[-1][ix,runner])
            widening=(p-backup)>=(history[-1][ix,leader]-history[-1][ix,runner]) - 1e-10
            t = TIMES[j-1]
            previous_t = 1 if j == 1 else TIMES[j-2]
            old_votes = fs[j-1][:,:,0]
            change = f[:,:,0]/t - old_votes/previous_t
            entrants = seen & (old_votes == 0) & nonempty
            entrant_share = (entrants * f[:,:,0]).sum(1)/t
            previous_change = np.where(nonempty[ix,prev_lead], change[ix,prev_lead], 0)
            rivals = (old_votes > 0) & nonempty
            rivals[ix,prev_lead] = False
            rival_gain = np.maximum(0, np.where(rivals, change, 0).max(1))
            conf = c[:,previous_t:t]
            entrant_response = (mem[:,previous_t:t] & entrants[:,None,:]).any(2)
            observed = entrant_response & np.isfinite(conf)
            count = observed.sum(1)
            total = np.where(observed, np.exp(np.minimum(conf,0)), 0).sum(1)
            train = split == "train"
            fill_conf = total[train].sum()/max(1,count[train].sum())
            entrant_conf = np.divide(total,count,out=np.full(q,fill_conf),where=count>0)
            states.append([p,p-backup,same&widening,drift,p-history[-1][ix,leader],backup-history[-1][ix,runner],previous_change,entrant_share,rival_gain,entrant_conf])
        prev_lead=leader;history.append(prob)
    s.update(zip(['p','gap','steady','drift','leader_change','runner_change','previous_change','entrant_share','rival_gain','entrant_conf'],np.array(states).transpose(1,2,0)))
    return s

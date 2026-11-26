"""Ablate confidence inputs to the joint-outcome model only. Keep the selector fixed."""
from expected_gain import replay
from latent_filter import observations,fit_filter
from joint_outcomes import joint_probabilities
from search import CACHE,OUT,TIMES
import pandas as pd
import numpy as np
names={262:'정오 공동 추정 · 득표만 사용',263:'정오 공동 추정 · 득표와 평균 확신도 사용',264:'정오 공동 추정 · 평균 확신도와 난이도 갱신에 확신도 사용',261:'정오 공동 추정 · 확신도 변화량까지 사용'}
rows=[]; scores=[]
for path in sorted(CACHE.glob('*.npz')):
    with np.load(path,allow_pickle=True) as f:s={k:f[k] for k in ['split','hit','valid','tok','margin','ids']}
    e,c,y,ids=observations(path);assert np.array_equal(ids,s['ids'])
    train=s['split']=='train';test=s['split']=='test'
    votes=fit_filter(e,c,y,train,False,0)[0];joint=fit_filter(e,c,y,train,True,0)[0]
    valid=np.isfinite(c);v=np.where(valid,np.exp(np.minimum(c,0)),0)
    fill=v[train].sum()/max(1,valid[train].sum())
    count=valid.cumsum(1);total=v.cumsum(1)
    def level(n):return np.divide(total[:,n-1],count[:,n-1],out=np.full(len(c),fill),where=count[:,n-1]>0)
    gains={k:np.zeros_like(s['hit'],float) for k in names}
    h=s['hit'][:,-1].astype(int)
    for j,t in enumerate(TIMES[:-1]):
        b=s['hit'][:,j].astype(int);m=s['margin'][:,j]/t;l=level(t)
        trend=l-level(t//2);trend[(count[:,t//2-1]==0)|(count[:,t-1]==count[:,t//2-1])]=0
        for k,cols in [(262,[m,votes[:,j]]),(263,[m,l,votes[:,j]]),(264,[m,l,joint[:,j]]),(261,[m,l,joint[:,j],trend])]:
            p=joint_probabilities(np.column_stack(cols),b,h,train);g=p[:,1]-p[:,2];gains[k][:,j]=g
            scores.append(pd.DataFrame(dict(id=k,cell=path.stem,q=ids[test],gain_error=(g[test]-(h-b)[test])**2,
                current_error=(p[test,2]+p[test,3]-b[test])**2)))
    for k,g in gains.items():
        for part in ['dev','test']:
            mask=s['split']==part
            sub={k:s[k][mask] for k in ['hit','valid','tok']};sub['gain']=g[mask]
            hit,n,tok=replay(sub,'gain',0)
            rows.append(pd.DataFrame(dict(id=k,kind=names[k],split=part,cell=path.stem,q=ids[mask],h=hit,n=n,t=tok,full=sub['tok'][:,-1])))
rows=pd.concat(rows)
tab=rows.groupby(['id','kind','split','cell']).agg(total=('h','size'),correct=('h','sum'),accuracy=('h','mean'),solutions=('n','mean'),tokens=('t','mean'),full_tokens=('full','mean')).reset_index()
tab['token_ratio']=tab.tokens/tab.full_tokens
tab.to_csv(OUT/'confidence_information_cells.csv',index=False)

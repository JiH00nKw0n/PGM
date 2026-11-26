"""Forecast the terminal maximum candidate score, then marginalize its threshold event."""
import numpy as np
import pandas as pd
from expected_gain import probability
from joint_outcomes import joint_probabilities
from latent_filter import observations,fit_filter
from search import BANK,CACHE,OUT,TIMES

rows=[];metrics=[]
for path in sorted(CACHE.glob('*.npz')):
    with np.load(path,allow_pickle=True) as f:s={k:f[k] for k in ['split','hit','valid','tok','margin','ids']}
    event,c,y,ids=observations(path)
    bank=BANK/(path.stem+'.parquet')
    raw=pd.read_parquet(bank,columns=['question_id','sample_index','prediction']).sort_values(['question_id','sample_index'])
    assert np.all(raw.question_id.to_numpy().reshape(-1,16)==ids[:,None]) and np.array_equal(s['ids'],ids)
    a=raw.prediction.fillna('').to_numpy().reshape(-1,16)
    train=s['split']=='train';test=s['split']=='test'
    rates=fit_filter(event,c,y,train,False,0)[0]
    slots=np.array([pd.factorize(row,sort=False)[0] for row in a])
    mem=(slots[:,:,None]==np.arange(16))&(a!='')[:,:,None]
    p=np.clip(np.exp(np.minimum(0,c)),.01,.99)
    weight=np.where(np.isfinite(c),np.log(p/(1-p)),0)
    totals=np.pad((mem*weight[:,:,None]).cumsum(1),((0,0),(1,0),(0,0)))
    seen=np.pad(mem.cumsum(1)>0,((0,0),(1,0),(0,0)))
    def maximum(mat,t):
        return np.where(seen[:,t].any(1),np.where(seen[:,t],mat,-np.inf).max(1),0.)
    h=s['hit'][:,-1].astype(int)
    terminal=maximum(totals[:,16],16)
    threshold=np.median(terminal[train&(h==1)])
    reached=(terminal>=threshold)&seen[:,16].any(1)
    buf={k:np.zeros_like(s['hit'],float) for k in [262,292,293,294]}
    for j,t in enumerate(TIMES[:-1]):
        level=maximum(totals[:,t],t)
        velocity=(totals[:,t]-totals[:,t-2])/2
        forecast=maximum(totals[:,t]+(16-t)*velocity,t)
        base=np.column_stack([s['margin'][:,j]/t,rates[:,j]])
        z=s['hit'][:,j].astype(int)
        for k in buf:
            x=base if k==262 else np.column_stack([base,level]) if k==292 else np.column_stack([base,level,forecast])
            # If the event does not vary on train, use the direct predictor.
            if k in (292,294) and np.unique(reached[train]).size>1:
                q=probability(x,reached.astype(int),train)
                probs=sum(w[:,None]*joint_probabilities(x,z,h,train&(reached==b)) for b,w in [(0,1-q),(1,q)])
                metrics.append((path.stem,k,t,threshold,((q[test]-reached[test])**2).mean()))
            else:probs=joint_probabilities(x,z,h,train)
            buf[k][:,j]=probs[:,1]-probs[:,2]
    for k,g in buf.items():
        for guard in [False,True]:
            number=(270 if guard else 262) if k==262 else k+3*guard
            desc={262:'득표만 사용',292:'현재 누적 점수로 16회 기준 도달 확률 추정',293:'후보별 점수 외삽으로 정오 변화 직접 추정',294:'후보별 점수 외삽으로 16회 기준 도달 확률 추정'}[k]
            ok=((g<=1e-12)|((s['margin']>=7)&guard))&s['valid'];ok[:,-1]=True
            end=ok.argmax(1)
            for part in ['dev','test']:
                ix=np.flatnonzero(s['split']==part)
                rows.append(pd.DataFrame(dict(id=number,kind=desc+(' · 7표 차에서도 중단' if guard else ' · 예상 이득 0 기준'),cell=path.stem,split=part,q=ids[ix],hit=s['hit'][ix,end[ix]],n=TIMES[end[ix]],tokens=s['tok'][ix,end[ix]],full=s['tok'][ix,-1])))
df=pd.concat(rows)
tab=df.groupby(['id','kind','split','cell']).agg(correct=('hit','sum'),total=('hit','size'),accuracy=('hit','mean'),solutions=('n','mean'),tokens=('tokens','mean'),full_tokens=('full','mean')).reset_index()
tab['token_ratio']=tab.tokens/tab.full_tokens
tab[tab.id>=292].to_csv(OUT/'projected_leader_cells.csv',index=False)
print(tab.groupby(['id','split'])[['accuracy','solutions']].mean().unstack().round(6))
m=pd.DataFrame(metrics,columns=['cell','id','t','threshold','brier'])
print(m.groupby('id').brier.mean(),m.cell.nunique())
m.to_csv(OUT/'projected_leader_prediction.csv',index=False)

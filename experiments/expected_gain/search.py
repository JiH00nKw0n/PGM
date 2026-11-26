"""Build candidate-selection states from the released response banks."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = Path(os.environ.get("PGM_RESULTS", Path(__file__).resolve().parent / "replay"))
BANK = Path(os.environ.get("PGM_BANKS", Path(__file__).resolve().parents[2] / "artifacts/banks"))
CACHE = OUT / "states"
TIMES = np.arange(2,17,2)
TOKEN_COLUMNS = ["generated_token_count","extraction_inline_output_tokens","extraction_final_output_tokens"]
COLS = ["split","question_id","sample_index","prediction","correct","answer_mean_logprob","answer_token_count",*TOKEN_COLUMNS]

def top_four(values):
    good = np.isfinite(values)
    weights = np.zeros(len(values))
    if good.sum() <= 4:
        weights[good] = 1
    else:
        cutoff = np.sort(values[good])[-4]
        weights[values > cutoff] = 1
        tied = values == cutoff
        weights[tied] = (4 - weights.sum()) / tied.sum()
    return weights

def prepare(cell):
    df = pd.read_parquet(BANK / (cell + ".parquet"),columns=COLS).sort_values(["question_id","sample_index"])
    ids = df.question_id.to_numpy()[::16]
    assert np.all(df.question_id.to_numpy().reshape(-1,16)==ids[:,None])
    assert np.all(df.sample_index.to_numpy().reshape(-1,16)==np.arange(16))
    split = df.split.to_numpy()[::16]
    train,dev = split=='train',split=='dev'
    ans = df.prediction.fillna('').to_numpy().reshape(-1,16)
    q = len(ids)
    correct = df.correct.to_numpy(bool).reshape(q,16)
    conf = df.answer_mean_logprob.where(df.answer_token_count>0).to_numpy().reshape(q,16)
    tok = df[COLS[-3:]].sum(axis=1).to_numpy().reshape(q,16).cumsum(1)[:,TIMES-1]
    # First appearance breaks equal scores, as in the earlier groupby(sort=False).
    slot = np.array([pd.factorize(row,sort=False)[0] for row in ans])
    members = slot[:,:,None]==np.arange(16)
    labels = (members & correct[:,:,None]).any(1)
    # Match the previous experiment's first response label for each answer.
    labels = np.take_along_axis(correct,members.argmax(1),axis=1)
    features=[]; margins=[]; maxcounts=[]; distinct=[]; confmean=[]
    for n in TIMES:
        mem=members[:,:n,:]
        votes=mem.sum(1)
        ac=conf[:,:n]
        finite=np.isfinite(ac)
        den=(mem*finite[:,:,None]).sum(1)
        mean=(mem*np.where(finite,ac,0)[:,:,None]).sum(1)/np.where(den,den,np.nan)
        top=(mem*np.array([top_four(row) for row in ac])[:,:,None]).sum(1)
        features.append(np.stack([votes,mean,top],axis=-1))
        ordered=np.sort(votes,axis=1)
        margins.append(ordered[:,-1]-ordered[:,-2])
        maxcounts.append(ordered[:,-1])
        distinct.append((votes>0).sum(1))
        confmean.append(np.sum(np.where(finite,ac,0),1)/np.maximum(finite.sum(1),1))
    full=features[-1]; present=full[:,:,0]>0
    use=train[:,None]&present
    fill=np.nan_to_num(np.nanmedian(full[use],axis=0))
    cleaned=[np.where(np.isfinite(f),f,fill) for f in features]
    scale=StandardScaler().fit(cleaned[-1][use])
    def outcomes(weights):
        ret=[]; nonempty=[]
        for f in cleaned:
            scores=np.einsum("qki,i->qk",(f-scale.mean_)/scale.scale_,weights)
            leader=np.where(f[:,:,0]>0,scores,-np.inf).argmax(1)
            ret.append(labels[np.arange(q),leader])
            nonempty.append(ans[np.arange(q),(slot==leader[:,None]).argmax(1)]!='')
        return np.stack(ret,1),np.stack(nonempty,1)
    w=np.broadcast_to(1/present.sum(1)[:,None],use.shape)[use]
    best=-1
    for c in [.01,.1,1,10]:
        lr=LogisticRegression(C=c,solver='liblinear',max_iter=1000).fit(scale.transform(cleaned[-1][use]),labels[use],sample_weight=w)
        hits,nonempty=outcomes(lr.coef_[0])
        score=hits[dev,-1].sum()
        if score>best:
            best=score; selected=hits,nonempty
    hits,nonempty=selected
    np.savez_compressed(CACHE/(cell+".npz"),ids=ids,split=split,hit=hits,valid=nonempty,tok=tok,margin=np.stack(margins,1))

if __name__ == "__main__":
    CACHE.mkdir(parents=True,exist_ok=True)
    for model in ["gemma4b","llama8b","qwen3b","qwen7b"]:
        for dataset in ["math","gsm8k","omnimath"]:
            prepare(f"{model}_{dataset}")

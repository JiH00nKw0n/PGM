"""Stop using expected change in the correctness of the submitted answer by response 16."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from latent_filter import observations,fit_filter
from search import CACHE,OUT,TIMES

def probability(x,y,rows):
    # Two pseudo-observations at the training mean regularize sparse conditional events.
    design=x[rows];target=y[rows]
    if len(target)==0: return np.full(len(x),.5)
    scale=StandardScaler().fit(design)
    design=np.vstack([scale.transform(design),np.zeros((2,x.shape[1]))])
    model=LogisticRegression(C=.25,solver='liblinear',max_iter=300).fit(design,np.r_[target,0,1])
    return expit(model.intercept_[0]+np.einsum('ij,j->i',scale.transform(x),model.coef_[0]))

def replay(s,key,threshold):
    hit=s['hit']; ix=np.arange(len(hit))
    stop=((s[key]<=threshold+1e-12)&s['valid']) if key!='ref' else ((s['margin']>=np.where(s['reg_group']==1,5,4))&s['valid'])
    stop[:,-1]=True;j=stop.argmax(1)
    return hit[ix,j],TIMES[j],s['tok'][ix,j]

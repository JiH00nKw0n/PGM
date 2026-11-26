import numpy as np
from scipy.special import softmax,logsumexp
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

def joint_probabilities(x,b,h,train):
    scale=StandardScaler().fit(x[train]);design=scale.transform(x)
    design=np.column_stack([design,np.ones(len(x))])
    A=np.vstack([design[train],np.tile(np.r_[np.zeros(x.shape[1]),1.],(4,1))])
    target=np.r_[(2*b+h)[train],np.arange(4)]
    def objective(flat):
        weights=flat.reshape(4,-1)
        logits=np.einsum('ni,ki->nk',A,weights)
        loss=(logsumexp(logits,axis=1)-logits[np.arange(len(A)),target]).sum()+2*(weights[:,:-1]**2).sum()
        err=softmax(logits,axis=1);err[np.arange(len(A)),target]-=1
        grad=np.einsum('nk,ni->ki',err,A);grad[:,:-1]+=4*weights[:,:-1]
        return loss,grad.ravel()
    fit=minimize(objective,np.zeros(4*A.shape[1]),jac=True,method='L-BFGS-B',options={'ftol':1e-11,'maxiter':500})
    assert fit.success
    logits=np.einsum('ni,ki->nk',design,fit.x.reshape(4,-1))
    return softmax(logits,axis=1)

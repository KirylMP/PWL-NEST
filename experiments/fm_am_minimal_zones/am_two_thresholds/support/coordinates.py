"""Normalized monotone mixture of power, exponential and logarithm."""
import numpy as np
REF=.5

def phi(c,shape):
    if shape is None:return np.asarray(c)/REF-1
    p,e,n,k=shape;t=np.asarray(c)/REF
    w=np.exp(np.array([p,e,0])-max(p,e,0));w/=w.sum()
    power=np.expm1(n*np.log(t))/n
    exponential=np.expm1(np.clip(k*(t-1),-700,700))/k if abs(k)>1e-7 else t-1
    return w[0]*power+w[1]*exponential+w[2]*np.log(t)

def dphi(c,shape):
    if shape is None:return np.ones_like(np.asarray(c))/REF
    p,e,n,k=shape;t=np.asarray(c)/REF
    w=np.exp(np.array([p,e,0])-max(p,e,0));w/=w.sum()
    return (w[0]*t**(n-1)+w[1]*np.exp(np.clip(k*(t-1),-700,700))+w[2]/t)/REF

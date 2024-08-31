import numpy as np
import pandas as pd
import itertools
from scipy.stats import binom
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from tqdm import tqdm
np.random.seed(10)


def f_c(p,b,n,alpha,beta):
    A = 8*(np.pi-3)/(3*np.pi*(4-np.pi))
    C = np.log(1-(2*alpha-1)**2)
    B = 4/np.pi + A*C
    wa = n* p - np.sqrt(n*p*(1-p)*(-B+np.sqrt(B**2-4*A*C))/A)
    return (wa*(1+b)-n+np.sqrt((n-wa*(1+b))**2-2*np.log(1+beta)*(n+wa*(b**2-1))))/(n+wa*(b**2-1))

#Simulation single bet
ps = [0.1,0.25,0.5,0.75,0.9]
bs = [9.5,3.15,1.05,0.35, 0.115]
ns = [100,250,500,1000]
alphas=[0.01, 0.025, 0.05, 0.1]
betas=[-0.01, -0.025, -0.05, -0.1]

pairs = [(ps[i],bs[i]) for i in range(len(ps))]
combs = [i for i in list(itertools.product(ps,bs,ns,alphas,betas)) if (i[0], i[1]) in pairs]

outcome = np.random.random((100000,max(ns)))

l_data = []

for (p,b,n,alpha,beta) in tqdm(combs):
    f = f_c(p,b,n,alpha,beta)
    returns = np.apply_along_axis(arr=outcome[:,:n], func1d=lambda rand: np.where(rand < p, 1+f*b, 1-f), axis=1)
    dist = np.cumprod(returns,axis=1).T[-1] - 1
    l_data.append([p,b,n,alpha,beta,f,np.mean(dist < beta)])

df = pd.DataFrame(l_data, columns=['p','b','n','alpha','beta','f','alpha_hat'])
df['pe'] = (df['alpha'] - df['alpha_hat']) / df['alpha']
df['ape'] = np.abs(df['alpha'] - df['alpha_hat']) / df['alpha']

df['alpha_hat'].corr(df['alpha'])
np.mean(df['alpha_hat']<df['alpha'])

for i in ps:
    dfi = df[df['p'] == i]
    np.mean(dfi['alpha_hat'] < dfi['alpha']), dfi['alpha_hat'].corr(dfi['alpha']), np.mean(np.abs(dfi['alpha_hat']-dfi['alpha']))


ps2 = [0.1,0.5,0.9]
combs = list(itertools.product(ps2,ns,alphas))
l2 = []
for p, n, alpha in combs:
    A = 8*(np.pi-3)/(3*np.pi*(4-np.pi))
    C = np.log(1-(2*alpha-1)**2)
    B = 4/np.pi + A*C
    wa = n* p - np.sqrt(n*p*(1-p)*(-B+np.sqrt(B**2-4*A*C))/A)
    l2.append([p,n,alpha,binom.cdf(np.floor(wa),n,p)])

df2 = pd.DataFrame(l2, columns=['p','n','alpha','floor_cdf'])
df2['diff'] = df2['floor_cdf'] - df2['alpha']

for i in ps2:
    dfi = df2[df2['p'] == i]
    np.mean(dfi['diff'])
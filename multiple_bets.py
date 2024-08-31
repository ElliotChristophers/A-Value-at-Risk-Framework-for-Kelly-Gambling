import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
np.random.seed(10)

def f_c_params(params,p,b,n,alpha,beta):
    A = 8*(np.pi-3)/(3*np.pi*(4-np.pi)) + params[0] + params[1] * p + params[2] * p**2 + params[3] * (p*(b+1)-1)
    C = np.log(1-(2*alpha-1)**2)
    B = 4/np.pi + A*C
    wa = n* p - np.sqrt(n*p*(1-p)*(-B+np.sqrt(B**2-4*A*C))/A)
    return (wa*(1+b)-n+np.sqrt((n-wa*(1+b))**2-2*np.log(1+beta)*(n+wa*(b**2-1))))/(n+wa*(b**2-1))

def sim(params, p_range, ev_range, n, alpha, beta):
    outcome = np.random.random((100000,n))
    ev = np.random.uniform(low=ev_range[0],high=ev_range[1],size=n)
    p = np.random.uniform(low=p_range[0], high=p_range[1], size=n)
    b = (ev+1)/p - 1
    f = np.zeros(n)
    for i in range(n):
        f[i] = f_c_params(params, p[i], b[i], n, alpha, beta)
    returns = np.apply_along_axis(arr=outcome, func1d=lambda rand: np.where(rand < p, 1+f*b, 1-f), axis=1)
    dist = np.cumprod(returns,axis=1).T[-1] - 1
    return np.mean(dist < beta)

n = 500
alpha = 0.05
beta = -0.05
ev_range = [0, 0.025]
p_range = [0.1,0.9]

theta_0 = np.linspace(-0.35,-0.55,5)
theta_1 = np.linspace(0.9,1.1,5)
theta_2 = np.linspace(1.5,1.65,5)
theta_3 = np.linspace(0.3,0.4,5)

x_params = list(itertools.product(theta_0, theta_1, theta_2, theta_3))

pss = np.linspace(p_range[0],p_range[1],9)
l = []
for params in tqdm(x_params):
    r = []
    for i in range(1,len(pss)):
        ri = []
        for j in range(5):
            ri.append(sim(params, [pss[i-1],pss[i]], ev_range, n, alpha, beta))
        #r.append(np.median(np.abs(np.array(ri)-alpha)))
        r.append(ri[list(np.abs(np.array(ri) - alpha)).index(np.median(np.abs(np.array(ri) - alpha)))])
    l.append([params, np.max(r)])


df = pd.DataFrame(np.array([list(i[:-1][0])+[i[-1]] for i in l]),columns=['theta_0','theta_1','theta_2','theta_3','r'])
df.sort_values(by='r').head(50)
params = [-0.4,0.933333,1.55,0.383333]
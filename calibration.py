import numpy as np
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
    
l=[]
for i in tqdm(range(1000)):
    p = np.random.rand()
    ev = np.random.uniform(low=0.005, high=0.05, size=1)[0]
    b = (ev+1)/p - 1
    n = round(10**np.random.uniform(low=2,high=3,size=1)[0])
    alpha = np.random.uniform(low=0.01,high=0.1,size=1)[0]
    beta = np.random.uniform(low=-0.1,high=-0.01,size=1)[0]
    f = f_c(p,b,n,alpha,beta)
    outcome = np.random.random((100000,n))
    returns = np.apply_along_axis(arr=outcome, func1d=lambda rand: np.where(rand < p, 1+f*b, 1-f), axis=1)
    dist = np.cumprod(returns,axis=1).T[-1] - 1
    l.append([p,np.abs(alpha-np.mean(dist < beta))])


p,d = np.array(sorted(l,key=lambda x: x[0])).T

window_width = 50
cumsum_vec = np.cumsum(np.insert(d, 0, 0)) 
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
p_adjusted = p[window_width//2-1: -window_width//2]

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(p_adjusted, ma_vec,color='0')
ax.set(xlabel=r'$p$',ylabel=r'$CMA_{\mid\hat{\alpha}-\alpha\mid}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
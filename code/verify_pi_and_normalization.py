from pathlib import Path
import importlib.util,json
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh
from verify_general_phi import coefficient,mod,OUT

import verify_B_nambu as B
results={"determinant":[],"pi_kernels":[],"normalization":[]}
for n in range(1,7):
    bonds=[.7+.08*j for j in range(n-1)]
    for E,g,mu in [(.37,.11,.43),(.83,.19,-.27)]:
        got=B.phase_ratio(n,E=E,Gamma=g,mu_M=mu,t_list=bonds)
        expected=16*(-1)**(n-1)*np.prod(np.array(bonds)**2)
        error=abs(got/expected-1)
        assert error<2e-8
        results['determinant'].append(dict(n=n,E=E,Gamma=g,mu=mu,relative_error=error))
for n in range(1,7):
    for resonant in [False,True]:
        q=1;mu=-2*np.cos(np.pi/(n+1)) if resonant else .37
        xi=mu+2*np.cos(np.arange(1,n+1)*np.pi/(n+1))
        scale=min(2,min(abs(xi[1:])) if resonant and n>1 else 2 if resonant else min(abs(xi)))
        rung=min(.08,scale/3)
        full=coefficient(n,rung,1,mu=mu)['coefficient']
        if resonant:
            a=np.r_[4.,4.,xi[1:]**2]
            weight=np.sqrt(2/(n+1))*np.sin(np.pi/(n+1))
            def f(w):
                x=w*w
                return .5*weight**4*(np.expm1(-np.log1p(x/a).sum())+x*(1/a).sum())/(x*x*np.pi)
        else:
            zero=np.prod(1/(xi*xi))
            def f(w):
                x=w*w
                return (zero*np.expm1(-2*np.log1p(x/4)-np.log1p(x/(xi*xi)).sum()))/(2*np.pi*x)
        kernel,error=quad(f,0,np.inf,epsabs=1e-7,epsrel=2e-8,limit=250)
        rel=abs(full/kernel-1)
        assert rel<2e-6,(n,resonant,full,kernel)
        results['pi_kernels'].append(dict(n=n,resonant=resonant,r=rung,full=full,kernel=kernel,relative_error=rel))
for n,q,r,s in [(1,1,.12,.5),(4,1,.12,.8),(4,2,.12,.5)]:
    c4=coefficient(n,r,s,q=q)['coefficient']
    gammas=r*np.array([.10,.07,.05])
    values=[]
    mu=-2*np.cos(np.pi*q/(n+1))+r*(s-1)
    for g in gammas:
        m=mod.KitaevSNS(L_left=10,L_right=10,L_N=n,mu_N=mu,mu_SC=0,Delta=1,t_perp=r,Gamma=g,phi=2*np.arcsin(s))
        en=[]
        for theta in [0,np.pi]:
            ev=eigh(m.BdG(theta),eigvals_only=True)
            en.append(-.25*np.abs(ev).sum())
        values.append(-(en[0]-en[1])/(2*g**4))
    extrap=float(np.polynomial.polynomial.polyfit(gammas**2,values,2)[0])
    rel=abs(extrap/c4-1)
    assert rel<5e-5,(n,q,rel)
    results['normalization'].append(dict(n=n,q=q,GF_fourth=c4,BdG_extrapolated=extrap,relative_error=rel))
(OUT/'pi-and-normalization-check.json').write_text(json.dumps(results,indent=2))
print(json.dumps({key:dict(count=len(vals),max_relative_error=max(a['relative_error'] for a in vals)) for key,vals in results.items()},indent=2))

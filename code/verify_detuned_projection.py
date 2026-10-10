from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh
from verify_general_phi import mod,OUT

def setup(r,s,eps):
    phi=2*np.arcsin(s)
    m=mod.KitaevSNS(L_left=3,L_right=3,L_N=1,mu_SC=0,mu_N=eps,Delta=1,t_perp=r,Gamma=0,phi=phi)
    H=m.BdG(0);n=H.shape[0]//2
    ref=mod.KitaevSNS(L_left=3,L_right=3,L_N=1,mu_SC=0,mu_N=0,Delta=1,t_perp=0,Gamma=0,phi=phi).BdG(0)
    columns=[]
    for region,site in [(0,2),(2,0)]:
        for leg in range(2):
            i=m.index(region,site,leg);alpha=leg*phi
            options=[]
            for sign in (1,-1):
                v=np.zeros(2*n,complex);v[i]=np.exp(.5j*alpha)/np.sqrt(2);v[i+n]=sign*np.exp(-.5j*alpha)/np.sqrt(2)
                options.append((np.linalg.norm(np.einsum('ij,j->i',ref,v,optimize=False)),v))
            err,v=min(options,key=lambda x:x[0]);assert err<1e-12
            columns.append(v)
    for leg in range(2):
        i=m.index(1,0,leg)
        for j in (i,i+n):
            v=np.zeros(2*n,complex);v[j]=1;columns.append(v)
    P=np.array(columns).T
    H8=np.einsum('ia,ij,jb->ab',P.conj(),H,P,optimize=False)
    def V8(theta):
        V=np.zeros_like(H)
        for leg in range(2):
            for i,j,right in [(m.index(0,2,leg),m.index(1,0,leg),False),(m.index(1,0,leg),m.index(2,0,leg),True)]:
                a=-np.exp(.5j*theta) if right else -1
                V[i,j]=a;V[j,i]=a.conjugate();V[i+n,j+n]=-a.conjugate();V[j+n,i+n]=-a
        return np.einsum('ia,ij,jb->ab',P.conj(),V,P,optimize=False)
    return H8,V8

def audit(r,s,eps):
    H,V=setup(r,s,eps)
    ev,U=eigh(H)
    Vs=[V(0),V(np.pi)]
    def trace_coefficient(w):
        G=np.einsum('ik,jk->ij',U/(1j*w-ev),U.conj(),optimize=False)
        traces=[]
        for v in Vs:
            a=np.einsum('ij,jk->ik',G,v,optimize=False)
            a2=np.einsum('ij,jk->ik',a,a,optimize=False)
            traces.append(np.einsum('ij,ji->',a2,a2,optimize=False).real)
        return -(traces[0]-traces[1])/(16*np.pi)
    got,err=quad(trace_coefficient,0,np.inf,epsabs=1e-8,epsrel=2e-8,limit=250)
    def proposed(w):
        numerator=w**4+(eps**2+r*r-7*r*r*s*s)*w*w+r*r*s*s*(r*r-eps*eps)
        denominator=(w*w+r*r*s*s)**2*(w*w+(eps+r)**2)*(w*w+(eps-r)**2)
        return numerator/(2*np.pi*denominator)
    kernel,_=quad(proposed,0,np.inf,epsabs=1e-8,epsrel=2e-8,limit=250)
    e=abs(eps)
    if e<r:
        closed=r*(1-s*s)*(1+3*s)/(4*s*(r*r*(1+s)**2-e*e)**2)
    else:
        closed=-(e*e*s+e*r*(5*s*s-1)+4*r*r*s**3)/(4*e*s*((e+r*s)**2-r*r)**2)
    rel=abs(got/closed-1) if closed else abs(got)
    assert abs(kernel-closed)<1e-6*max(1,abs(closed))
    assert rel<1e-7,(r,s,eps,got,closed)
    return dict(r=r,s=s,epsilon=eps,trace_kernel=got,proposed_kernel=kernel,closed=closed,relative_error=rel)

if __name__=='__main__':
    results=[audit(.12,s,eps) for s in (.2,1/3,.5,.8,1.) for eps in (-.3,-.06,0.,.06,.3,.6)]
    (OUT/'detuned-projection-check.json').write_text(json.dumps(results,indent=2))
    print('Independent 8-dimensional projection:',len(results),'cases; max error',max(x['relative_error'] for x in results))

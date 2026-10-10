"""Independent fourth-order audit from the full uncoupled finite BdG resolvent.

The interface is gauged rather than projecting onto the eight Majoranas.
All trace operations are restricted exactly to the finite interface support.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad

BASE = Path(__file__).resolve().parent.parent
OUT = BASE/'generated/checks'
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('sns', BASE/'code/general_kitaev_sns.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def coefficient(n, r, s, q=1, lead=10, mu=None, delta=1):
    if mu is None:
        mu = -2*np.cos(np.pi*q/(n+1)) + r*(s-1)
    m = mod.KitaevSNS(L_left=lead, L_right=lead, L_N=n,
        mu_SC=0, mu_N=mu, Delta=delta, t_perp=r, Gamma=0,
        phi=2*np.arcsin(s))
    H = m.BdG(0)
    assert np.max(np.abs(H-H.conj().T)) < 1e-13
    energies, U = eigh(H)
    pairs = []
    for b in range(2):
        pairs.append((m.index(0,lead-1,b),m.index(1,0,b),False))
        pairs.append((m.index(1,n-1,b),m.index(2,0,b),True))
    npart = H.shape[0]//2
    particle = sorted({i for i,j,k in pairs}|{j for i,j,k in pairs})
    inds = particle+[i+npart for i in particle]
    loc = {v:i for i,v in enumerate(inds)}
    W = U[inds,:]
    def vmat(theta):
        V=np.zeros((len(inds),len(inds)),complex)
        for i,j,right in pairs:
            a=-np.exp(0.5j*theta) if right else -1
            V[loc[i],loc[j]]=a
            V[loc[j],loc[i]]=np.conj(a)
            V[loc[i+npart],loc[j+npart]]=-np.conj(a)
            V[loc[j+npart],loc[i+npart]]=-a
        return V
    Vs=[vmat(0),vmat(np.pi)]
    def integrand(omega):
        G=np.einsum('ik,jk->ij',W/(1j*omega-energies),W.conj(),optimize=False)
        values=[]
        for V in Vs:
            A=np.einsum('ij,jk->ik',G,V,optimize=False)
            A2=np.einsum('ij,jk->ik',A,A,optimize=False)
            values.append(np.einsum('ij,ji->',A2,A2,optimize=False).real)
        return -(values[0]-values[1])/(16*np.pi)
    value, error = quad(integrand,0,np.inf,epsabs=2e-7,epsrel=2e-8,limit=300)
    weight=(2/(n+1))**0.5*np.sin(np.pi*q/(n+1))
    prediction=(1-s*s)*(1+3*s)/(64*s**3)
    scaled=r**3*value/weight**4
    return dict(n=n,q=q,r=r,s=s,lead=lead,coefficient=value,
        quadrature_error=error,scaled=scaled,prediction=prediction,
        relative_difference=(scaled/prediction-1) if prediction else None,
        uncoupled_gap=float(min(abs(energies))))

if __name__=='__main__':
    results=[]
    for n in range(1,5):
        for r in (0.12,0.06,0.03):
            for s in (0.5,0.8):
                a=coefficient(n,r,s)
                results.append(a)
                print(json.dumps(a),flush=True)
    for n in range(1,5):
        a=coefficient(n,0.12,1)
        results.append(a); print(json.dumps(a),flush=True)
    for lead in (4,8,16):
        a=coefficient(4,0.06,0.5,lead=lead)
        results.append(a); print(json.dumps(a),flush=True)
    for n in range(2,7):
        for q in range(1,n+1):
            a=coefficient(n,0.01,0.5,q=q,lead=8)
            results.append(a); print(json.dumps(a),flush=True)
    (OUT/'general-phi-check.json').write_text(json.dumps(results,indent=2))

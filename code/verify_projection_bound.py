from verify_general_phi import mod,OUT
from scipy.linalg import eigh
import numpy as np,json
from pathlib import Path
# Compare the exact nearby spectral subspace with its zero-coupling limit.
# Remote end Majoranas are excluded by maximizing contact support.
results=[]
for lead in (3,6,10):
 for s in (.5,.8):
  for r in (.08,.04,.02):
   models=[mod.KitaevSNS(L_left=lead,L_right=lead,L_N=1,mu_SC=0,mu_N=rr*(s-1),Delta=1,t_perp=rr,Gamma=0,phi=2*np.arcsin(s)) for rr in (0,r)]
   hs=[m.BdG(0) for m in models];m=models[0];np_=hs[0].shape[0]//2
   contact=[]
   for reg,x in ((0,lead-1),(1,0),(2,0)):
    for b in (0,1):
     i=m.index(reg,x,b);contact.extend((i,i+np_))
   bases=[]
   for h in hs:
    ev,u=eigh(h);u=u[:,abs(ev)<.5]
    vals,rot=eigh(np.einsum('ia,ib->ab',u[contact].conj(),u[contact],optimize=False));basis=np.einsum('ia,ab->ib',u,rot[:,np.argsort(vals)[-8:]],optimize=False);bases.append(basis)
   p0,p1=bases
   # Polar unitary aligns nearby subspaces with positive mutual overlap.
   left,sv,right=np.linalg.svd(np.einsum('ia,ib->ab',p0.conj(),p1,optimize=False));aligned=np.einsum('ia,ba,cb->ic',p1,right.conj(),left.conj(),optimize=False)
   h8=np.einsum('ia,ij,jb->ab',p0.conj(),hs[1],p0,optimize=False);exact=np.einsum('ia,ij,jb->ab',aligned.conj(),hs[1],aligned,optimize=False)
   pr=np.linalg.norm(np.einsum('ia,ja->ij',p0,p0.conj(),optimize=False)-np.einsum('ia,ja->ij',p1,p1.conj(),optimize=False),2)
   herr=np.linalg.norm(exact-h8,2)
   vals=[]
   for theta in (0,1.2,np.pi):
    v=np.zeros_like(hs[0]);npart=v.shape[0]//2
    for b in (0,1):
     for i,j,isr in ((m.index(0,lead-1,b),m.index(1,0,b),False),(m.index(1,0,b),m.index(2,0,b),True)):
      a=-np.exp(.5j*theta) if isr else -1
      v[i,j]=a;v[j,i]=a.conjugate();v[i+npart,j+npart]=-a.conjugate();v[j+npart,i+npart]=-a
    vals.append(np.linalg.norm(np.einsum('ia,ij,jb->ab',aligned.conj(),v,aligned,optimize=False)-np.einsum('ia,ij,jb->ab',p0.conj(),v,p0,optimize=False),2))
   verr=max(vals)
   results.append(dict(lead=lead,s=s,r=r,projection_norm=pr,hamiltonian_error=herr,interface_error=verr,scaled_projection=pr/r,scaled_hamiltonian=herr/r**2,scaled_interface=verr/r))
# These are representative numerical checks of the estimates, not a replacement for proof.
assert max(x['scaled_projection'] for x in results)<1
assert max(x['scaled_hamiltonian'] for x in results)<1
assert max(x['scaled_interface'] for x in results)<1
(OUT/'projection_bound_check.json').write_text(json.dumps(results,indent=2))
print('18 representative checks passed; scaled norms:',*[max(x[k] for x in results) for k in ('scaled_projection','scaled_hamiltonian','scaled_interface')])

import sympy as sp
import numpy as np
import json
from pathlib import Path

(Path(__file__).resolve().parent.parent/'generated/checks').mkdir(parents=True,exist_ok=True)
g,s,c=sp.symbols('g s c',real=True)
def matrix(theta):
    A=sp.zeros(8)
    def set_(i,j,v):A[i,j]=v;A[j,i]=-v
    set_(0,1,s);set_(2,3,s)
    set_(4,5,s-1);set_(6,7,s-1);set_(4,7,-1);set_(6,5,-1)
    if theta==0:
        cosL=cosR=[1,c];sinL=sinR=[0,s]
    else:
        rt=sp.sqrt(2)
        cosL=[1/rt,(c+s)/rt];sinL=[-1/rt,(s-c)/rt]
        cosR=[1/rt,(c-s)/rt];sinR=[1/rt,(c+s)/rt]
    for b in range(2):
        a=4+2*b;bb=5+2*b
        set_(b,a,g*cosL[b]);set_(b,bb,-g*sinL[b])
        set_(2+b,a,-g*sinR[b]);set_(2+b,bb,-g*cosR[b])
    return A
def reduce_c(expr):
    return sp.factor(sp.Poly(sp.expand(expr),c).rem(sp.Poly(c*c+s*s-1,c)).as_expr())
char=[]
for th in [0,sp.pi]:
    A=matrix(th)
    cp=A.charpoly().all_coeffs()
    ek=[reduce_c(cp[2*k]) for k in range(1,5)]
    char.append(ek)
    print('theta',th,'e1/e2/e3/e4',ek,flush=True)

# Elementary symmetric relations of four positive energies.
S0=[2*(1+s),6*s,2*s*s*(3-s),s**3*(2-s)]
x=sp.symbols('x')
a1,a2,a3,b1,b2,b3=sp.symbols('a1 a2 a3 b1 b2 b3')
solutions=[]
for k,ek in enumerate(char):
    # Pfaffian has constant sign below the crossing threshold.
    sig4=S0[3]-2*x*s*s+x*x*(1 if k==0 else 0)
    sig1=S0[0]+a1*x+b1*x*x
    sig2=S0[1]+a2*x+b2*x*x
    sig3=S0[2]+a3*x+b3*x*x
    exprs=[sig1**2-2*sig2-ek[0].subs(g**2,x),
        sig2**2-2*sig1*sig3+2*sig4-ek[1].subs(g**2,x),
        sig3**2-2*sig2*sig4-ek[2].subs(g**2,x)]
    sol2=sp.solve([sp.expand(e).coeff(x,1) for e in exprs],[a1,a2,a3],dict=True)[0]
    sol4=sp.solve([sp.expand(e.subs(sol2)).coeff(x,2) for e in exprs],[b1,b2,b3],dict=True)[0]
    sol2={v:sp.factor(z) for v,z in sol2.items()};sol4={v:sp.factor(z) for v,z in sol4.items()}
    print('theta',k,'order2',sol2,'order4',sol4,flush=True)
    solutions.append((sol2,sol4))
c4=sp.factor((solutions[0][1][b1]-solutions[1][1][b1])/2)
expected=(1-s*s)*(1+3*s)/(32*s**3)
assert sp.simplify(c4-expected)==0
print('cos(theta) coefficient of positive energy sum:',c4,flush=True)
out={'second_order_sum':str(solutions[0][0][a1]),'fourth_order_cos_sum':str(c4),
     'fourth_order_constant_sum':str(sp.factor((solutions[0][1][b1]+solutions[1][1][b1])/2)),
     'squared_energy_invariants_theta0':[str(z) for z in char[0]],
     'squared_energy_invariants_thetapi':[str(z) for z in char[1]],
     'energy_to_J1_factor':'1/2', 'symbolic_cos_coefficient_verified':True}
(Path(__file__).resolve().parent.parent/'generated/checks/low_coefficients_check.json').write_text(json.dumps(out,indent=2))

from pathlib import Path
import hashlib, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter, FixedLocator
from matplotlib.patches import Patch
from general_kitaev_sns import KitaevSNS

BASE = Path(__file__).resolve().parent.parent
OUT = BASE/'generated'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'STIXGeneral', 'mathtext.fontset':'stix',
    'font.size':9, 'axes.labelsize':10, 'axes.titlesize':10,
    'xtick.labelsize':9, 'ytick.labelsize':9, 'legend.fontsize':8.5,
    'axes.linewidth':.7, 'lines.linewidth':1.2, 'pdf.fonttype':42,
    'xtick.major.size':3, 'ytick.major.size':3, 'axes.labelpad':2})
audit = {'font_sizes_pt': {'axis':10,'ticks':9,'legend':8.5,'inset':7.5},
         'original_sha256':{}, 'raster_viewports':{}}
for f in (BASE/'figures').glob('*'):
    if f.is_file(): audit['original_sha256'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()

def save(fig, name):
    # Fixed physical width: LaTeX will reproduce these sizes without shrinking.
    fig.savefig(OUT/(name+'.pdf'), dpi=600)
    fig.savefig(OUT/(name+'.png'), dpi=220)
    plt.close(fig)

def image_rgb(name):
    im=Image.open(BASE/'figures'/name).convert('RGBA')
    bg=Image.new('RGBA',im.size,'white'); bg.alpha_composite(im)
    return np.array(bg.convert('RGB'))

# Fig. 1 remains unchanged, as requested by the author.
# Fig. 2: retain original numerical phase-map pixels, re-typeset annotations.
# No phase boundaries are inferred or recomputed. A pre-existing legend is
# replaced only inside its original annotation rectangle.
im=image_rgb('currentphase_1.png')
rois=[(313,225,1931,1404),(2344,225,3962,1404)]
audit['raster_viewports']['currentphase_1.png']=rois
fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
fig.subplots_adjust(left=.075,right=.99,bottom=.18,top=.9,wspace=.25)
for k,(ax,roi) in enumerate(zip(axes,rois)):
    x0,y0,x1,y1=roi; data=im[y0:y1+1,x0:x1+1].copy()
    if k==1:
        # Old legend bbox in source image: 2703..3605, 254..575.
        # White background remains an explicitly visible annotation box.
        data[21:352,357:1265]=255
    ax.imshow(data,extent=(-.15,3.15,-.1,2.1),aspect='auto',interpolation='none')
    ax.set(xlim=(-.15,3.15),ylim=(-.1,2.1),xlabel=r'$\mu/t$',ylabel=r'$t_\perp/t$')
    ax.set_xticks([0,1,2,3]);ax.set_yticks([0,1,2])
    ax.set_title(f'({chr(97+k)}) '+r'$L_{\rm M}='+str(4+k)+'$',loc='left')
    if k==1:
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-.15+357/1618*3.3,2.1-352/1179*2.2),908/1618*3.3,331/1179*2.2,facecolor='white',edgecolor='.8',lw=.6,zorder=3))
        handles=[Patch(facecolor=c,label=l) for c,l in [('green',r'Mixed topological--$\pi$'),('mediumorchid',r'$\pi$-junction'),('yellow',r'$0$-junction'),('blue','Topological')]]
        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.221,.982),
                  frameon=False,borderpad=.12,labelspacing=.02,
                  handlelength=1.2,handletextpad=.5,fontsize=9)
save(fig,'fig2_phases_largefont')

# Original recursive-Green-function CPR samples are available only as images.
# Preserve all original blue sample pixels. Only non-blue frame/grid pixels
# are erased from the viewport and replaced with native vector axes/grid.
# This is annotation re-layout, not numerical digitization or a new calculation.
def cpr(name,rois,scales,factors,outname):
    im=image_rgb(name); fig,axes=plt.subplots(2,2,figsize=(3.4,2.85))
    fig.subplots_adjust(left=.19,right=.99,bottom=.135,top=.925,wspace=.72,hspace=.73)
    audit['raster_viewports'][name]={'roi_edges':rois,'display_ymax':scales,'factors':factors}
    for i,(ax,roi,mx,fac) in enumerate(zip(axes.flat,rois,scales,factors)):
        x0,y0,x1,y1=roi; data=im[y0-5:y1+6,x0-14:x1+15].copy()
        # Exact RGB data preservation, including samples at viewport boundaries.
        blue=(data[:,:,2].astype(int)>data[:,:,0].astype(int)+10)&(data[:,:,2].astype(int)>data[:,:,1].astype(int)+10)
        data[~blue]=255
        ax.imshow(data,extent=(-2-14*4/(x1-x0),2+14*4/(x1-x0),-mx-5*2*mx/(y1-y0),mx+5*2*mx/(y1-y0)),aspect='auto',interpolation='none',zorder=2,clip_on=False)
        ax.set(xlim=(-2,2),ylim=(-mx,mx),xlabel=r'$\theta/\pi$',ylabel=r'$J(\theta)/t$')
        ax.set_xticks([-2,0,2]); ax.set_yticks([-mx,0,mx]);ax.set_yticklabels([f'{-mx:g}','0',f'{mx:g}'])
        ax.axhline(0,c='.75',lw=.45,ls=':',zorder=3);ax.axvline(0,c='.75',lw=.45,ls=':',zorder=1)
        ax.set_title(f'({chr(97+i)})',loc='left',pad=3)
        if fac: ax.text(1,1.03,rf'$\times10^{{{fac}}}$',transform=ax.transAxes,ha='right',fontsize=9)
    save(fig,outname)
cpr('l4mu0.png',[(431,136,1643,1027),(2291,136,3502,1023),(446,1360,1657,2254),(2293,1360,3504,2249)],
    [.006,.004,.006,.006],[0,0,0,-5],'fig3_off_cpr_largefont')
cpr('l5mu1.png',[(458,93,1667,996),(2304,96,3526,1009),(440,1359,1662,2257),(2312,1382,3536,2297)],
    [.01,.01,.01,.01],[0,0,-1,0],'fig7_res_cpr_largefont')

# Fig. 4(a-c): exact original cached scans, checked against the full mu grids.
cache=np.load(BASE/'data/fig5_J1_cache.npz',allow_pickle=False)
def scan(mus,phi,gamma):
    payload={'L_N':5,'t':1.,'Delta':.6,'t_perp':.1,'mu_min':float(mus[0]),'mu_max':float(mus[-1]),'mu_n':len(mus),'phi':float(phi),'Gamma':float(gamma),'theta_n':401}
    key=hashlib.sha1(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
    assert np.array_equal(cache['mus_'+key],mus),key
    return cache['J1_'+key].copy()
ma=np.linspace(-.5,1.5,161)
mc=np.unique(np.r_[np.linspace(-.5,1.5,81),np.linspace(-.12,.08,31),np.linspace(.88,1.08,31)])
phis=[np.pi,2*np.pi/3,np.pi/2,np.pi/3]
ja=scan(ma,np.pi/3,.1); jc=[scan(mc,p,.1) for p in phis]
gs=np.linspace(.025,.065,6); jg=np.array([scan(np.array([0.]),np.pi,g)[0] for g in gs])
slope=np.polyfit(np.log(gs),np.log(jg),1)[0]
assert np.all(jg>0) and np.all(np.sqrt(2)*gs<.1)
audit['fourth_order_fit']=float(slope)

# Fig. 4(d): same BdG model, theta grid, and physical parameters as original.
# Correct the old mismatch between a theta=-pi baseline and theta=0 label.
theta=np.linspace(-np.pi,np.pi,401)
energy_file=BASE/'data/energy_curve_raw.npz'
if energy_file.exists():
    en=np.load(energy_file); eo=en['raw_off']; er=en['raw_res']
else:
    energies=[]
    for mu in [.3,0.]:
        model=KitaevSNS(L_N=5,mu_N=mu,mu_SC=0.,t_perp=.1,Gamma=.1,phi=np.pi/3,Delta=.6,t=1.)
        energies.append(np.array([model.ground_energy(th) for th in theta]))
    eo,er=energies
    np.savez(energy_file,theta=theta,raw_off=eo,raw_res=er)
do=eo-eo[200]; dr=er-er[200]
assert do[200]==0 and dr[200]==0
assert np.allclose(np.diff(do),np.diff(eo-eo[0]),atol=1e-13)
audit['energy_baseline']='E_GS(theta)-E_GS(0), corrected from original actual theta=-pi subtraction'
audit['energy_min_theta_pi']=[float(theta[np.argmin(do)]/np.pi),float(theta[np.argmin(dr)]/np.pi)]
np.savez(OUT/'figure4_data.npz',mu_a=ma,mu_c=mc,J1_a=ja,J1_c=np.array(jc),phi=np.array(phis),Gamma=gs,J1_inset=jg,theta=theta,energy_off=do,energy_res=dr)

fig,axes=plt.subplots(1,4,figsize=(7.1,2.58),gridspec_kw={'width_ratios':[1,1,1.25,1]})
fig.subplots_adjust(left=.063,right=.992,bottom=.32,top=.84,wspace=.58)
titles=['(a) Alignment','(b) First harmonic','(c) Phase dependence','(d) Josephson energy']
for ax,title in zip(axes,titles): ax.set_title(title,loc='left',fontsize=9.5,pad=12)
ax=axes[0]
for q,c in [(3,'#1f77b4'),(4,'#ff7f0e')]:
    eps=ma+.1+2*np.cos(np.pi*q/6); align=.049-.1-2*np.cos(np.pi*q/6)
    ax.plot(ma,eps,c=c,label=rf'$q={q}$');ax.plot(align,.049,'ko',ms=3)
    ax.axvline(align,c='.65',ls='--',lw=.65)
ax.axhline(.049,c='k',ls=':',lw=.8,label=r'$E_{\rm end}$')
ax.set(xlim=(-.5,1.5),ylim=(-1.5,1.6),xlabel=r'$\mu_{\rm M}/t$',ylabel=r'$\epsilon^{\rm M}/t$')
ax.set_xticks([0,1]);ax.set_yticks([-1,0,1]);ax.legend(loc='lower right',frameon=False,fontsize=8.5,handlelength=1.2,labelspacing=.1)
ax=axes[1];ax.plot(ma,ja*1e3,c='k')
for q in [3,4]:ax.axvline(.049-.1-2*np.cos(np.pi*q/6),c='.6',lw=.65,ls='--')
ax.axhline(0,c='.6',lw=.65);ax.set(xlim=(-.5,1.5),xlabel=r'$\mu_{\rm M}/t$',ylabel=r'$J_1/t\ (10^{-3})$')
ax.set_xticks([0,1]);ax.yaxis.set_major_locator(MaxNLocator(3))
ax=axes[2]
for y,l in zip(jc,[r'$\phi=\pi$',r'$\phi=2\pi/3$',r'$\phi=\pi/2$',r'$\phi=\pi/3$']):ax.plot(mc,y*1e3,label=l)
ax.axhline(0,c='.6',lw=.65);ax.set(xlim=(-.5,1.5),xlabel=r'$\mu_{\rm M}/t$',ylabel=r'$J_1/t\ (10^{-3})$')
ax.set_xticks([0,1]);ax.yaxis.set_major_locator(MaxNLocator(3))
handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=((ax.get_position().x0+ax.get_position().x1)/2,.002),frameon=False,ncol=2,fontsize=8,handlelength=.8,columnspacing=.5,labelspacing=.08,handletextpad=.3)
ins=ax.inset_axes([.43,.62,.25,.36]);ins.loglog(gs,jg,'ko-',ms=2,lw=.7)
ins.set_xticks([.03,.06]);ins.set_xticklabels(['.03','.06'],fontsize=7.5)
ins.set_yticks([1e-7,1e-6]);ins.set_yticklabels([r'$10^{-7}$',r'$10^{-6}$'],fontsize=7.5)
ins.xaxis.set_minor_formatter(NullFormatter());ins.yaxis.set_minor_formatter(NullFormatter())
ins.tick_params(which='both',length=1.5,pad=.5)
ins.set_xlabel(r'$\Gamma/t$',fontsize=7.5,labelpad=0)
ins.set_title(rf'$p={slope:.2f}$',fontsize=7.5,pad=2)
ax=axes[3]; ax.plot(theta/np.pi,do*1e3,c='#1f77b4',label=r'$\mu_{\rm M}/t=0.3$');ax.plot(theta/np.pi,dr*1e3,c='#d62728',label=r'$\mu_{\rm M}/t=0$')
ax.axhline(0,c='.6',lw=.6);ax.set(xlim=(-1,1),xlabel=r'$\theta/\pi$',ylabel=r'$\Delta E_{\rm GS}/t\ (10^{-3})$')
ax.set_xticks([-1,0,1]);ax.yaxis.set_major_locator(MaxNLocator(3))
handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=((ax.get_position().x0+ax.get_position().x1)/2,.002),frameon=False,fontsize=8,handlelength=1.1,labelspacing=.05)
save(fig,'fig4_resonance_energy_largefont')

# Fig. 5 (previous Fig. 6): regenerate the identical analytic formulas.
env={};source=(BASE/'code/make_fig7_clamping_v4.py').read_text()
exec(source.split('# Figure layout:')[0],env)
th=env['theta']/np.pi
fig=plt.figure(figsize=(7.1,3.8))
g=fig.add_gridspec(2,3,left=.095,right=.99,bottom=.13,top=.81,wspace=.48,hspace=.19,height_ratios=[1.2,1])
tops=[fig.add_subplot(g[0,i]) for i in range(2)]
bottoms=[fig.add_subplot(g[1,i]) for i in range(2)]
for i,tag in enumerate(['off','res']):
    ax=tops[i]
    for key,c,ls,l in [('Em1','#1f77b4','-',r'$e_-+t_\perp$'),('Em2','#1f77b4','--',r'$|e_--t_\perp|$'),('Ep1','#ff7f0e','-',r'$e_++t_\perp$'),('Ep2','#ff7f0e','--',r'$|e_+-t_\perp|$')]:
        ax.plot(th,env[key+'_'+tag],c=c,ls=ls,label=l)
    ax.axhline(.16,c='.45',ls=':',lw=.8)
    ax.set(xlim=(0,2),ylim=(0,.355));ax.set_xticks([0,1,2]);ax.tick_params(labelbottom=False)
    ax.set_title(['(a) Off resonance','(b) Resonance'][i],loc='left',pad=35)
    if i==0:ax.set_ylabel(r'$E/t$')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.32),ncol=2,frameon=False,fontsize=7.5,handlelength=1,columnspacing=.6,labelspacing=.08)
    ax=bottoms[i]
    ax.plot(th,env['calEm_'+tag],c='#1f77b4',lw=1.8,label=r'$\mathcal{E}_-$')
    ax.plot(th,env['calEp_'+tag],c='#ff7f0e',lw=1.3,ls='--' if i else '-',label=r'$\mathcal{E}_+$')
    ax.set(xlim=(0,2),xlabel=r'$\theta/\pi$',ylim=(-.176,-.144));ax.set_xticks([0,1,2]);ax.set_yticks([-.17,-.16,-.15])
    if i==0:ax.set_ylabel(r'$\mathcal{E}/t$')
    ax.legend(loc='lower right',frameon=False,fontsize=8,handlelength=1.2)
    ax.text(.5,.86,['Selective clamping','Collective clamping'][i],transform=ax.transAxes,ha='center',fontsize=8.5)
ax=fig.add_subplot(g[:,2]); eps=np.linspace(0,.3,600); t1,t2,t3=env['thresholds'](eps,.1)
for a,b,c in [(0,t1,'#1f77b4'),(t1,t2,'#ff7f0e'),(t2,t3,'#2ca02c'),(t3,.35,'#777777')]:ax.fill_between(eps,a,b,color=c,alpha=.12)
for y,c,l in [(t1,'#1f77b4',r'$t_1$'),(t2,'#ff7f0e',r'$t_2$'),(t3,'#2ca02c',r'$t_3$')]:ax.plot(eps,y,c=c,label=l)
ax.set(xlim=(0,.3),ylim=(0,.35),xlabel=r'$|\varepsilon|/t$',ylabel=r'$t_\perp/t$');ax.set_xticks([0,.1,.2,.3]);ax.set_yticks([0,.1,.2,.3]);ax.set_title('(c) Detuning map',loc='left',pad=12)
for x,y,txt in [(.19,.025,'Branch switching'),(.14,.14,'One branch active\n'+r'(smooth-$\pi$)'),(.255,.264,'Partial\nclamping'),(.18,.328,'Fully clamped')]:ax.text(x,y,txt,ha='center',va='center',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
ax.legend(loc='center right',bbox_to_anchor=(1,.35),frameon=False,fontsize=8)
save(fig,'fig5_clamping_largefont')

# Fig. 6 (previous Fig. 7): exact classification on the original 801x801 grid.
mu=np.linspace(0,3,801);tp=np.linspace(0,3,801);MU,TP=np.meshgrid(mu,tp)
w=(np.abs(MU+TP)<2).astype(int)+(np.abs(MU-TP)<2).astype(int)
from matplotlib.colors import ListedColormap,BoundaryNorm
fig,axes=plt.subplots(1,2,figsize=(3.4,1.85));fig.subplots_adjust(left=.15,right=.99,bottom=.23,top=.85,wspace=.35)
for i,(ax,z,cols) in enumerate(zip(axes,[w,w%2],[['white','#1786b8','#ffe000'],['white','#1786b8']])):
    ax.imshow(z,origin='lower',extent=(0,3,0,3),interpolation='nearest',cmap=ListedColormap(cols),vmin=-.5,vmax=len(cols)-.5,aspect='auto',rasterized=True)
    ax.set(xlim=(0,3),ylim=(0,3),xlabel=r'$\mu_{\rm S}/t$');ax.set_xticks([0,1,2,3]);ax.set_yticks([0,1,2,3])
    if i==0:ax.set_ylabel(r'$t_\perp/t$')
    ax.set_title(['(a) '+r'$\phi=0$','(b) '+r'$0<\phi<\pi$'][i],loc='left',fontsize=9.5)
    for x,y,txt in ([(.5,.5,r'$w=2$'),(1.45,2.35,r'$w=1$'),(2.5,.25,r'$w=0$')] if i==0 else [(.5,.5,r'$\nu=0$'),(1.45,2.35,r'$\nu=1$'),(2.5,.25,r'$\nu=0$')]):
        ax.text(x,y,txt,ha='center',fontsize=9)
save(fig,'fig6_topology_largefont')
(OUT/'figure_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print('All six revised publication assets generated; Fig. 1 preserved. Fourth-order slope:',slope)

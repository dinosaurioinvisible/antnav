

import numpy as np
from auxs import *
# from source.utils import display_image, animated_window, plot_map, squash_deg, rotate
# from source.display import plot_3d, plot_route
from pdb import set_trace as pbx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

navi, route, ims, qims, rxyo, qxys = load_navi_and_route(navi_type='seq2seq',
                                                         route_type='basic',
                                                         route_id = 0,
                                                         navi_corrs=False)

# ds (80,360 : 28,800; dx=5 -> 16,72 : 1152; dx=4 -> 20,90 : 1800, dx=8 -> 10,45 : 450)
def mk_dst(ii, dx=4):
    vp,hp = (np.array(ii.shape[1:])/dx).astype(int)
    dst = np.zeros((ii.shape[0],vp,hp))
    for pi in range(vp):
        di = pi*dx
        for pj in range(hp):
            dj = pj*dx
            dst[:,pi,pj] = np.sum(ii[:,di:di+dx,dj:dj+dx],axis=(1,2))/(dx*dx)
    return dst
ims = mk_dst(ims)
qims = mk_dst(qims)

# difs
iid = ims[:-1] - ims[1:]
iix = np.abs(iid)
# iipd = np.where(iid>0,iid,0)
# iind = np.where(iid<0,iid,0)
# masks
iip = np.where(iid>0,1,0)
iin = np.where(iid<0,1,0)   # mk pos, dif channels
# pos, neg difs & comb
iipx = iix * iip        # same iipd, keep for masking
iinx = iix * iin        # same as iind, same (sign changes)
# iicx = iipx - iinx    # same as iid!

def animplot(iims, rows=2, cols=2, step=250):
    fig, axs = plt.subplots(rows,cols, figsize=(10,5))
    ti = 0
    ims = []
    for ei,ax in enumerate(axs.flat):
        im = ax.imshow(iims[ei][ti], cmap='gray', aspect='auto', animated=True)
        ims.append(im)
    def update_fig(ti):
        ti = (ti+1)%iims[0].shape[0]
        for ui,ax in enumerate(axs.flat):
            ims[ui].set_array(iims[ui][ti])
        return [im for im in ims]
    ani = animation.FuncAnimation(fig,update_fig,interval=step,blit=True,repeat=True)
    plt.show()
    plt.close()
# animplot([ims[1:],iix,iipx,iinx], step=100)
    
# la pregunta es como se puede guardar una memoria minima de un evento
# que luego permita su reconstruccion
# como en el caso de las palabras desordenadas:
# debe haber un partidor por saliencia (1a, ultima y (desord) letras en ese caso)
# un tejedor algoritmico fijo (tiene que aplicar a todo caso)
# y un terminador (basicamente por capacidad cognitiva)











# qiid = qims[:-1] - qims[1:]
# qiix = np.abs(qiid)
# qiip = np.where(qiid>0,1,0)
# qiin = np.where(qiid<0,1,0)
# qiipx = qiix * qiip
# qiinx = qiix * qiin


# def mk_sxt(ii, ws=11, th=8):
#     sxt, swt = [], []
#     ii = np.where(ii>0,1,0)
#     for bw in range(ii.shape[0]-ws):
#         fw = bw + ws
#         mw = np.sum(ii[bw:fw],axis=0)
#         mt = np.where(mw>=th,1,0)
#         sxt.append(mt)
#         # sw = np.zeros((mw.shape))
#         # for thi in range(1,ws):
#         #     sw = sw + np.where(mw==thi,mw,0) * thi/ws
#         # swt.append(sw)
#     return np.array(sxt) #, np.array(swt)
# psx = mk_sxt(iip)
# nsx = mk_sxt(iin)
# qpsx = mk_sxt(qiip, ws=2, th=2)
# pnsx = mk_sxt(qiin, ws=2, th=2)




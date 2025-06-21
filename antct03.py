
import numpy as np
from auxs import *
from pdb import set_trace as pbx 
import matplotlib.pyplot as plt 

navi, route, ims, qims, rxyo, qxys = load_navi_and_route(navi_type='seq2seq',
                                                         route_type='basic',
                                                         route_id = 0,
                                                         navi_corrs=False)

# difs 
iid = ims[:-1] - ims[1:]
iib = np.abs(iid)
# normalized sums
# iids = iid.sum(axis=(1,2))/28800
# iibs = iib.sum(axis=(1,2))/28800

def mk_difs(ims,mw=5,as_events=True):
    ii_dif = ims[:-1] - ims[1:]
    ii_dif = ii_dif.clip(-1,1) if as_events else ii_dif

        


        

# iid = np.where(iid<0,-1,0) + np.where(iid>0,1,0)
# iib = np.where(iib>0,1,0)

# we need to choose only 2, one for abs difs & one for temp contrast,
# for structural and temporal dimensions of object
# clotho, lachesis, atropos

iid2 = iid[:-1] - iid[1:]
iib2 = np.abs(iib[:-1] - iib[1:])

iid3 = iid2[:-1] - iid2[1:]
iib3 = np.abs(iib2[:-1] - iib2[1:])

iid4 = iid3[:-1] - iid3[1:]
iib4 = np.abs(iib3[:-1] - iib3[1:])

iidxs = iid4.sum(axis=(1,2))/28800
iibxs = iib4.sum(axis=(1,2))/28800
plt.plot(iidxs)
plt.plot(iibxs)
plt.show()
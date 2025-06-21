
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
iids = iid.sum(axis=(1,2))/28800
iibs = iib.sum(axis=(1,2))/28800

# plots difs
plt.plot(iids)
plt.plot(iibs)
# plt.show()

# difs of higher order
iids2 = iids[:-1] - iids[1:]
iibs2 = np.abs(iibs[:-1] - iibs[1:])
# plt.plot(iids2)
# plt.plot(iibs2)

iids3 = iids2[:-1] - iids2[1:]
iids4 = np.zeros(ims.shape[0])
# iids4[2:-2] = iids3[:-1] - iids3[1:]
iids4[4:] = iids3[:-1] - iids3[1:]
# plt.plot(iids4/5)

idx = np.abs(iids4).astype(int)
plt.plot(np.abs(idx))

vi = False
th = 10
th_end = 0
for ei,i in enumerate(idx):
    if not vi:
        if i > th:
            plt.axvline(ei-2,color='gray')
            print()
            print(ei,i)
            vi = True
    else:
        print(ei,i)
        if i <= th_end:
            plt.axvline(ei+2,color='black')
            vi = False

plt.axhline(th, color= 'purple')
plt.show()

imsd2 = iid[:-1] - iid[1:]
imsd3 = imsd2[:-1] - imsd2[1:]
imsd4 = ims.copy()
# imsd4[2:-2,:,:] = imsd3[:-1] - imsd3[1:]
imsd4[4:] = imsd3[:-1] - imsd3[1:]          # to init with non identical vals
# animplot([imsd4,ims],step=100,color='viridis')

# temporal contrasts
imdx = imsd4.copy()
imdx[:4] = 0 
iip = np.where(imdx > 0, 1, 0)
iin = np.where(imdx < 0, -1, 0)
# animplot([iip,ims[1:]])

iips = np.sum(iip, axis=(1,2))/28800
iins = np.sum(iin, axis=(1,2))/28800
# plt.plot(iips)
# plt.plot(iins)

iisx = iips + iins
# plt.plot(iisx)
plt.plot(np.abs(iisx))      # abs after dif (temporal contrast) for each dif

# plt.plot(np.abs(idx/255))   # 0:255 -> -1:1, abs val before dif

iibs3 = np.abs(iibs2[:-1] - iibs2[1:])
iibs4 = np.abs(iibs3[:-1] - iibs3[1:])
iibx = np.zeros(idx.shape[0])
iibx[4:] = iibs4/255            # abs before difs, for every dif
plt.plot(iibx)

plt.show()




import numpy as np
from auxs import *
from pdb import set_trace as pp
import matplotlib.pyplot as plt
from source.utils import rotate

# how many timesteps does it take to give form to an 'image' that can be 'compared'
# given a stream of events, how long does it take to get a salient cluster
# saliency and temporality should be inversely proportional
# more info (more saliency) would make mental lapses shorter
# more info (more cluttered, less saliency) would make mental lapses to be longer
# cog. speaking, primitive reps in memory and in working memory would have an intrinsic temporal dimension
# because memories are made from working memory, 
# if the function of working memory is to enable/facilitate cognitive distinctions (by salient features)
# so that these salient features enable identification by association
# then memories should be as sparse/filtered as possible, i.e. extracted from background as possible
# experimental idea then
# make memories as representations of salient feature constructions, from navigation
# compare to rotational constructions of salient features
# because, given that there are no snapshots really, both need to be build up 
# and because nevigational and rotational are different in escence, 
# the (only?) way they could be matched
# is by converting the input streams into the most basic feature rep possible


navi, route, ims, qims, rxyo, qxys = load_navi_and_route(navi_type='seq2seq',
                                                         route_type='basic',
                                                         route_id = 0,
                                                         navi_corrs=False)

# difs
# ii_difs = ims[:-1] - ims[1:]
# ii_pos = np.where(ii_difs > 0, 1, 0)
# ii_neg = np.where(ii_difs < 0, 1, 0)
# plt.plot(ii_pos.sum(axis=(1,2)))
# plt.plot(ii_neg.sum(axis=(1,2)))
# plt.show()
# pos and neg are basically the same, although a bit delayed

# idx = 35
# mw = 20
# th = 1
# dx_th = 80 * 360 / 15
# # ii_norm = ii_difs/255
# # ii_abs = np.abs(ii_difs)
# mask = ii_pos[idx] + ii_neg[idx]
# mask_th = mask.nonzero()[0].shape[0] + 80 * 360 / 15
# dx = np.zeros((80,360))
# for i in range(1,mw+1):
#     # dx += ii_pos[idx+i] * i/mw
#     # dx += ii_pos[idx+i] * i/mw - ii_neg[idx+i] * i/mw
#     # dx += ii_difs[idx+i] * i/mw                           # good, but relaying on 0:255 intensities
#     # iidx = ii_pos[idx+i+1] - ii_pos[idx+i]
#     # dx += np.where(iidx>0,iidx,0) * i/mw
#     # dx += ii_pos[idx+i] - ii_neg[idx+i]
#     # dx += ii_pos[idx+i] * np.where(mask==0,1,0) - ii_neg[idx+i] * np.where(mask==0,1,0)
#     if dx.nonzero()[0].shape[0] > dx_th or mask.nonzero()[0].shape[0] > mask_th:
#         break
#     dx += ii_pos[idx+i] * np.where(mask==0,i/mw,0) - ii_neg[idx+i] * np.where(mask==0,i/mw,0)
#     mask += ii_pos[idx+i] + ii_neg[idx+i]
#     print(f'\n{idx+i}/{idx+mw}')
#     print(f'dx nonzeros = {dx.nonzero()[0].shape[0]} / {dx_th}')
#     print(f'mask nonzeros = {mask.nonzero()[0].shape[0]} / {mask_th}')
# # dx = np.where(np.abs(dx) > th, np.abs(dx), 0)
# # dx = np.where(np.abs(dx) > th, dx, 0)
# plt.imshow(dx,aspect='auto')
# plt.show()


def mk_mem(ims,overlap=2,mi=0,mf=0,max_mw=20,th=15,fix_mw=False):
    # turn images into 'event-like' data
    ii_dx = ims[:-1] - ims[1:]
    ii_pos = np.where(ii_dx > 0, 1, 0)
    ii_neg = np.where(ii_dx < 0, 1, 0)
    # check params
    max_mw = max_mw if ii_dx.shape[0] >= max_mw else ii_dx.shape[0]     # max frames
    idx = np.max([0,mi]).clip(max=ii_dx.shape[0]-max_mw)                # starting id
    mf = mf if mf >= idx + max_mw else ii_dx.shape[0] - max_mw          # last mw starting id
    overlap = overlap if overlap <= max_mw/2 else int(max_mw/10)        # id overlap for integration
    # mk integrated frames
    mx_th = ims.shape[1] * ims.shape[2] / th
    mx_ids = []
    mx_ims = []
    # for the whole image set
    while idx < mf:

        idf = idx + 1
        # mask = ii_dx[idx]
        pmask = ii_pos[idx]
        nmask = ii_neg[idx]
        # mask_th = mask.nonzero()[0].shape[0] + mx_th
        mx = np.zeros((ii_dx.shape[1:]))                    # integrated mem frame
        mx_sum = 0
        
        # for every possible max window
        for i in range(1,max_mw+1):
            if not fix_mw:
                mx_sum = np.sum(np.abs(mx))
                print(mx_sum)
                # if mx.nonzero()[0].shape[0] > mx_th or mask.nonzero()[0].shape[0] > mask_th:
                if mx_sum > 80 * 360 / th:
                    break
            idf = idx + i
            # mx += ii_pos[idx+i] * np.where(mask==0,i/10,0) - ii_neg[idx+i] * np.where(mask==0,i/10,0)
            # mx += ii_pos[idx+i] * np.where(mask==0,i/max_mw,0) - ii_neg[idx+i] * np.where(mask==0,i/max_mw,0)
            # mx += ii_pos[idx+i] * np.where(pmask==0,i/10,0) - ii_neg[idx+i] * np.where(nmask==0,i/10,0)
            mx += ii_pos[idx+i] * np.where(pmask==0,i,0) - ii_neg[idx+i] * np.where(nmask==0,i,0)
            # mask += ii_dx[idx+i]
            pmask += ii_pos[idx+i]
            nmask += ii_neg[idx+i]
        
        wlen = idf - idx
        mx /= wlen
        mx_ids.append([idx,idf])
        mx_ims.append(mx)
        print(f'\nidx: {idx}, idf: {idf}, len: {idf-idx}, max id: {idx+max_mw}')
        print(f'dx nonzeros = {mx.nonzero()[0].shape[0]} / {mx_th}')
        # print(f'mask nonzeros = {mask.nonzero()[0].shape[0]} / {mask_th}')
        # plt.imshow(mx,aspect='auto')
        # plt.show()
        plot_imgs([mx, mx.clip(0), np.abs(mx), np.where(mx<=0.5,0,mx)])
        idx = idf - overlap
        
    return np.array(mx_ims), np.array(mx_ids)

# wims, wids = mk_mem(ims, overlap=1, mi=0, mf=60, max_mw=25, th=1, fix_mw=False)

def mk_rot_img(im, rdegs=10, th=1):
    rims = []
    for deg in range(0,360,rdegs):
        rim = rotate(deg, im)
        rims.append(rim)
    rims = np.array(rims)
    rdifs = rims[:-1] - rims[1:]
    ii_pos = np.where(rdifs > 0, 1, 0)
    ii_neg = np.where(rdifs < 0, 1, 0)
    rx = np.zeros((80,360))
    idx = 0
    # rx = ii_pos[idx] - ii_neg[idx]
    pmask = ii_pos[idx]
    nmask = ii_neg[idx]
    for i in range(1,rdifs.shape[0]):
        rx += ii_pos[idx+i] * np.where(pmask==0,1,0) - ii_neg[idx+i] * np.where(nmask==0,1,0)
        # rx += ii_pos[idx+i] * np.where(pmask==0,i+1,0) - ii_neg[idx+i] * np.where(nmask==0,i+1,0)
        pmask += ii_pos[idx+i]
        nmask += ii_neg[idx+i]
    rx = rx / rdifs.shape[0]
    plot_imgs([im, np.abs(rx), rx.clip(0), np.where(np.abs(rx)<=0.5,0,rx)])
    return rx, rdifs

rim, rdifs = mk_rot_img(ims[44])
# rim, rdifs = mk_rot_img(ims[100])
# rim, rdifs = mk_rot_img(ims[202])

# ii_sums = iidx.sum(axis=(1,2))/255      # normalized intensities
# ii_sums_abs = np.abs(ii_sums)           # abs value (no contrast)

# 1. difs vs abs difs
# plt.plot(ii_sums)
# plt.plot(ii_sums_abs)
# plt.show()
# animplot([iidx,np.abs(iidx)])
# mostly the same re time location

# ii_pos = np.where(iidx>0,1,0)           # ~ pos events
# ii_neg = np.where(iidx<0,-1,0)          # ~ neg events
# evs = ii_pos + ii_neg

# 2. pos evs vs neg evs
# plt.plot(ii_pos.sum(axis=(1,2)))
# plt.plot(ii_neg.sum(axis=(1,2)))
# plt.plot(evs.sum(axis=(1,2)))
# plt.show()
# animplot([evs,np.abs(evs)])
# basically the same (neg a bit delayed (~1))

# 3. difs vs events (events are 0/1 difs)
# plt.plot(ii_sums)
# plt.plot(evs.sum(axis=(1,2)))
# plt.show()
# ~same, but evs diffs are higher in critical points
# also they are more informative in general

# 4. difs vs events 
# animplot([iidx,evs])
# evs lose a lot of resolution

# why ants get disoriented when moved?
# ants need time to make an event/working-mem window
# mem windows lengths are variable according to info
# comparisons for loc/action are among wm and memories
# assumptions:
# a) memories are made from working memory windows
# b) working mem windows encode working mem objects
# c) wmem-memory matching is based on object encoding
# questions (adapted to dataset):
# how many pm frames make a working mem window?
# how much info (objects) can the wmem encode?
# how does rotations influence this process?

# 5. compare subsequent difs
# evs3 = evs[:-1] - evs[1:]
# evs4 = evs3[:-1] - evs3[1:]
# evs5 = evs4[:-1] - evs4[1:]
# evs6 = evs5[:-1] - evs5[1:]
# evs7 = evs6[:-1] - evs6[1:]

# plt.plot(evs.sum(axis=(1,2)))
# plt.plot(evs7.sum(axis=(1,2))/7)
# plt.show()
# animplot([evs,evs3])
# animplot([evs,evs7])
# animplot([ims,evs,evs3,evs7])
# further difs produce new intensities-like difs
# landscape seems richer because of temp contrast

# 6. get and visualize only high dif events
# esum = evs.sum(axis=(1,2)).astype(int)
# threshold = 1000
# xids = np.where(np.abs(esum)<threshold)[0][1:]  # 1: for imshow
# evx = evs.copy()
# evx[xids] = 0
# animplot([ims,evx])
# mostly occlusions

# 7. temporal kernel 
# instead of the scalar given by the sliding tensor




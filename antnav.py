
import numpy as np
from copy import deepcopy
from auxs import *
from tqdm import tqdm
from source.utils import *
import pdb
import matplotlib.pyplot as plt
from source.display import *

# questions:
# when init SPP, Pipeline() is applied to route images, or only to query? # line 104 seqnav
# 

navi_type = 'spm'
route_type = 'basic'
route_id = 1

navi,rx,rx_imgs,qx_imgs = load_navi_and_route(navi_type=navi_type,
                                              route_type=route_type,
                                              route_id=route_id)
ims = np.array(rx_imgs)
qims = np.array(qx_imgs)
rxyo,qxys = get_corr_xy_points(rx)

print_data = True
plots = False

idx,hix = 0,0
# wblim,wflim = 0,0
dq_bl_id,dq_fl_id = None,None
dq_bl_hd,dq_fl_hd = None,None


for ei,qxim in enumerate(qx_imgs):
    print('\n{}/{}'.format(ei,len(qx_imgs)))

    if True:
        wblim = navi.blimit
        wflim = navi.flimit
    else: 
        wblim = max(0, idx-10)
        wflim = min(idx+11, ims.shape[0])

    if 5 <= ei: # <= len(qx_imgs)-5:
        ws = 3
        bfs = []
        dqs = []
        dq_id_sums = []
        dq_hd_sums = []
        qw = qims[ei-ws+1:ei+1]
        rw = ims[wblim:wflim]
        for wi in range(wflim-wblim+1):
            # rwi = ims[navi.blimit+wi:navi.blimit+wi+3]
            rwi = ims[wblim+wi:wblim+wi+ws]
            dq = rwi - qw
            # bfs.append((navi.blimit+wi,navi.blimit+wi+3))
            bfs.append((wblim+wi,wblim+wi+ws))
            dqs.append(dq)
            dq_id_sums.append(np.sum(dq))
            dq_hd_sums.append(np.sum(np.abs(dq)))
        
        dqs = np.array(dqs)
        dq_id_sums = np.array(dq_id_sums)
        dq_id_min = np.min(np.abs(dq_id_sums))
        dq_id = int(np.where(np.abs(dq_id_sums)==dq_id_min)[0])

        dq_hd_sums = np.array(dq_hd_sums)
        dq_hd_min = np.min(np.abs(dq_hd_sums))
        dq_hd = int(np.where(np.abs(dq_hd_sums)==dq_hd_min)[0])
        
        dq_bl_id,dq_fl_id = bfs[dq_id]
        wsums = np.abs(np.sum(dqs[dq_id],axis=(1,2)))
        idx = dq_bl_id + int(np.where(wsums==np.min(wsums))[0])
        
        # heading
        dq_bl_hd,dq_fl_hd = bfs[dq_hd]
        # hix = navi.get_heading_ix(qxim,navi.route_images[idx-5:idx+6])
        # print(hix)
        hix = navi.get_heading_ix(qxim,navi.route_images[dq_bl_id-ws:dq_fl_id+ws])
        # print(hix)
        # hix = navi.get_heading_ix(qxim,navi.route_images[idx-2:idx+1])
        # print(hix) 

        # r0
        # 48, 48,   48, 48, 48, 46, 
        # 49, 42, -100, 61, 43, 48, [idx-5 : idx+6]         -> 11
        # 49, 47, -100, 59, 42, 48, [dq_bl-3 : dq_fl+3]     -> 9 
        # 80, 47,   37, 59, 12, 48, [idx-2 : idx+1]         -> 3

        # pdb.set_trace()

    # save window
    navi_blim = navi.blimit
    navi_flim = navi.flimit
    # correlative route/mem img
    rx_id = int(qxys[ei][2])
    rx_yaw = rxyo[rx_id][2]
    rx_img = rx_imgs[rx_id]
    # get heading and mp idx
    heading = navi.get_heading(qxim)
    mp = navi.mem_pointer
    # navi matching heading query img
    navi_qx_img = rotate(heading,qxim)
    # navi mem pointer img
    navi_mp_yaw = rxyo[mp][2]
    navi_mp_img = rx_imgs[mp]
    # ix matching heading query img
    ix_qx_img = rotate(hix,qxim)
    # ix mem pointer img
    ix_mp_yaw = rxyo[idx][2]
    ix_mp_img = rx_imgs[idx]

    if print_data:
        print()
        print(ei)
        print('id pointer: navi = {}, route = {}, ix = {}'.format(mp,rx_id,idx))
        print('headings: navi = {}, route = {}, ix = {}'.format(heading,round(rx_yaw,3),hix))
        print('navi window: {}:{}, ix id window: {}:{}, ix heading window: {}:{}'.format(navi_blim,navi_flim,dq_bl_id,dq_fl_id,dq_bl_hd,dq_fl_hd))

    # pdb.set_trace()

    # if True:
    # if np.abs(heading-rx_yaw)>20 or np.abs(mp-rx_id)>10: # or np.abs(idx-rx_id)>10 or np.abs(hix-rx_heading)>20:
    if np.abs(mp - rx_id) > 10: # or (ei > 5 and np.abs(idx - rx_id) > 10):
        print('\nid mismatch\n')
        print(ei)
        print('id pointer: navi = {}, route = {}, ix = {}'.format(mp,rx_id,idx))
        print('headings: navi = {}, route = {}, ix = {}'.format(heading,round(rx_yaw,3),hix))
        print('navi window: {}:{}, ix id window: {}:{}, ix heading window: {}:{}'.format(navi_blim,navi_flim,dq_bl_id,dq_fl_id,dq_bl_hd,dq_fl_hd))
        if plots:
            # query img & route correct img & yaw
            # navi rotated query img, navi mp idx & yaw
            # ix rotated query img, ix mp idx img & yaw
            subs = [
                    'query img: {}, route/mem id: {}, heading: {}'.format(ei,rx_id,0),
                    'correlative img, route/mem id: {}, yaw: {}'.format(rx_id,round(rx_yaw,3)),
                    'navi match rot qx img, heading: {}'.format(heading),
                    'navi mp img, window: {}:{}, idx: {}, yaw: {}'.format(navi_blim,navi_flim,mp,round(navi_mp_yaw,3)),
                    'ix match rot qx img, window: {}:{}, heading: {}'.format(wblim,wflim,hix),
                    'ix mp img, segm: {}:{}, idx: {}, yaw: {}'.format(dq_bl_id,dq_fl_id,idx,round(ix_mp_yaw,3))
                    ]
            plot_imgs([qxim, rx_img,
                    navi_qx_img, navi_mp_img,
                    ix_qx_img, ix_mp_img],
                    rows=3,cols=2,
                    subtitles=subs)
            # plot_3d(navi.wrsims)
            # plot_3d(navi.wrsims_ix)
        # reset pointer
        navi.update_mid_pointer(rx_id - navi.blimit)
        wblim = navi.blimit
        wflim = navi.flimit

        pdb.set_trace()
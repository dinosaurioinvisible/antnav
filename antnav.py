
import numpy as np
from auxs import *
from tqdm import tqdm
from source.utils import display_image, animated_window, plot_map, squash_deg, rotate
from source.display import plot_3d, plot_route
from pdb import set_trace as pbx
import matplotlib.pyplot as plt

# doubts:
# spm gets 3 ids & 22 headings mismatches, but seq2seq gets 11 ids & 49 hds? (r0)
# similar for other routes it seems, maybe is the preprocessing?

# navi + rxyo, qxys; rx and qx imgs as tensor arrays
# navi types: 'spm' or 'seq2seq'
navi, route, ims, qims = load_navi_and_route(navi_type= 'seq2seq',
                                            route_type= 'basic',
                                            route_id = 0)

def test(navi,qims,route, perfect_window=False, plot_xhs=False):
    # lists for mismatches
    msm_nxi,msm_nxh = [],[]
    msm_uxi,msm_uxh = [],[]

    for qid,qim in enumerate(qims):
        print()
        print(qid)

        rid = int(navi.qxys[qid][2])
        rhx = int(navi.rxyo[rid][2])
        if perfect_window==True or abs(navi.mem_pointer-rid) >= navi.window:
            navi.reset_window(rid)
        navi_hx = navi.get_heading(qim)
        navi_mp = navi.mem_pointer
        
        if qid >= 2:
            if perfect_window:
                navi.reset_window(rid)
            if qid <= 2:
                navi.window_blim = navi.blimit
                navi.window_flim = navi.flimit
            else:
                wbl,wfl = navi.mk_mem_window(qim)
                print('window: {}:{}, navi window: {}:{}'.format(wbl,wfl,navi.blimit,navi.flimit))
            
            query_seq = qims[qid-2:qid+1]
            ix_mp, ix_hx = navi.get_heading_ix(query_seq)
            navi.append_data(qid)
            
            print(rid,navi_mp,ix_mp)
            print(rhx,navi_hx,ix_hx)

            if abs(rid-navi_mp) >= 10: 
                msm_nxi.append([qid,rid,navi_mp,ix_mp,rhx,navi_hx,ix_hx])
            if abs(rhx-navi_hx) > 10:
                msm_nxh.append([qid,rid,navi_mp,ix_mp,rhx,navi_hx,ix_hx])
            if abs(rid-ix_mp) >= 10: 
                msm_uxi.append([qid,rid,navi_mp,ix_mp,rhx,navi_hx,ix_hx])
            if abs(rhx-ix_hx) > 10:
                msm_uxh.append([qid,rid,navi_mp,ix_mp,rhx,navi_hx,ix_hx])

    print('\nnavi msms: ids: {}, hds: {}, ix msms: ids: {}, hds: {}\n'.format(len(msm_nxi),len(msm_nxh),len(msm_uxi),len(msm_uxh)))

    # mismatching headings; xh: common, uxh: only integrated, nxh: only navigator
    xh = np.array([hx for hx in msm_uxh if hx[0] in np.array(msm_nxh)[:,0]])
    uxh = np.array([ux for ux in msm_uxh if ux[0] not in np.array(msm_nxh)[:,0]])
    nxh = np.array([nx for nx in msm_nxh if nx[0] not in np.array(msm_uxh)[:,0]])
    # normally few mismatching ids for perfect mem windows (for tests)
    # xh: common, uxi: integrated (maybe also in nxi), nxi: navi (maybe also in uxi)
    xi = np.array([x for x in msm_uxi if x[0] in np.array(msm_nxi)[:,0]])
    uxi = np.array(msm_uxi)
    nxi = np.array(msm_nxi)

    if plot_xhs:
        print()
        for ux in uxh:
            print('qx id {}, rx id: {}, navi id: {}, idx: {}, rx yaw: {}, navi hx: {}, ix hx: {}'.format(*zip(ux)))
            if abs(ux[4] - ux[6]) > 20:
                print('\nlarge heading/yaw difference\n')
                idx_yaw = navi.rxyo[ux[3]][2]
                plot_imgs([qims[ux[0]], ims[ux[1]], rotate(ux[6],qims[ux[0]]), ims[ux[3]]],
                        subtitles = ['query img, id: {}'.format(ux[0]),
                                    'route corr img, id: {}, yaw: {}'.format(ux[1],ux[4]),
                                    'rotated query img, heading: {}'.format(ux[6]),
                                    'chosen mem img, id: {}, yaw: {}'.format(ux[3],int(idx_yaw))])
    plot_route(route.route_dict,traj=navi.route_data)

    return navi, xh,uxh,nxh, xi,uxi,nxi
# navi, xh,uxh,nxh, xi,uxi,nxi = test(navi,qims,route)

# TODO: run exps and save data in dataframe or whatever
for ri in range(7):
    navi, route, ims, qims = load_navi_and_route(navi_type= 'seq2seq',
                                            route_type= 'basic',
                                            route_id = ri)
    navi, xh,uxh,nxh, xi,uxi,nxi = test(navi,qims,route)

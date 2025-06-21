
import numpy as np
from auxs import *
from source.utils import display_image, animated_window, plot_map, squash_deg, rotate
from source.display import plot_3d, plot_route
from pdb import set_trace as pbx
import matplotlib.pyplot as plt

# doubts:
# spm gets 3 ids & 22 headings mismatches, but seq2seq gets 11 ids & 49 hds? (r0)
# similar for other routes it seems, maybe is the preprocessing?

# navi + rxyo, qxys; rx and qx imgs as tensor arrays
# navi types: 'spm' or 'seq2seq'
navi, route, ims, qims, rxyo, qxys = load_navi_and_route(navi_type='seq2seq', route_type='basic', route_id = 0, navi_corrs=False)



def test_navi(route_id=0,queue_size=3,sub_window=9):
    navi, route, ims, qims, rxyo, qxys = load_navi_and_route(navi_type='seq2seq',
                                                             route_type='basic',
                                                             navi_corrs=False,
                                                             route_id=route_id,
                                                             queue_size=queue_size,
                                                             sub_window=sub_window)

    ixs, hxs = [], []
    for qid,qim in enumerate(qims):
        rid = int(qxys[qid][2])
        yaw = int(rxyo[rid][2])
        # print('\nqid: {}, rid: {}'.format(qid,rid))
        hix = navi.get_heading(qim)
        print('qid: {}, rid: {} -> mp: {}, yaw: {} -> hx: {}\n'.format(qid,rid,navi.mem_pointer,yaw,hix))
        
        if abs(yaw - hix) > 15:
            print('angle mismatch')
            print('qid: {}, rid: {} -> mp: {}, yaw: {} -> hx: {}\n'.format(qid,rid,navi.mem_pointer,yaw,hix))
            hxs.append([qid,rid,navi.mem_pointer,yaw,hix])

        if abs(rid - navi.mem_pointer) > 20:
            print('id mismatch')
            print('qid: {}, rid: {} -> mp: {}, yaw: {} -> hx: {}\n'.format(qid,rid,navi.mem_pointer,yaw,hix))
            ixs.append([qid,rid,navi.mem_pointer,yaw,hix])
            # reset id
            navi.mem_pointer = rid
            navi.reset_window(navi.mem_pointer)
    
    print('id errors: {}'.format(len(ixs)))
    print('heading errors: {}'.format(len(hxs)))
    # save_simdata(hxs)
    return navi,ims, np.array(ixs), np.array(hxs)

# navi,ims, ixs,hxs = test_navi(route_id=0,
#                               queue_size=5,
#                               sub_window=9)




def test(route_id=0,route_type='basic',navi_type='spm',perfect_window=False,mk_plots=False,save_data=True):

    navi, route, ims, qims = load_navi_and_route(navi_type=navi_type,route_type=route_type,route_id=route_id)
    
    # data from mismatches & dict
    msm_nxi,msm_nxh = [],[]
    msm_uxi,msm_uxh = [],[]
    datakeys = ['qid','rid','mp','idx','yaw','heading','hix']
    simdata = dict([(dkey,[]) for dkey in datakeys])

    for qid,qim in enumerate(qims):
        
        print('\n{}/{}'.format(qid,qims.shape[0]-1))

        rid = int(navi.qxys[qid][2])
        yaw = int(navi.rxyo[rid][2])
        if perfect_window==True or abs(navi.mem_pointer-rid) >= navi.window:
            navi.reset_window(rid)
        navi_hx = navi.get_heading(qim)
        navi_mp = navi.mem_pointer
        
        if qid >= 2:
            if perfect_window:
                navi.reset_window(rid)
            # if qid <= 2:
            #     navi.window_blim = navi.blimit
            #     navi.window_flim = navi.flimit
            # else:
            #     wbl,wfl = navi.mk_mem_window(qim)
            #     print('window: {}:{}, navi window: {}:{}'.format(wbl,wfl,navi.blimit,navi.flimit))
            
            query_seq = qims[qid-2:qid+1]
            ix_mp, ix_hx = navi.get_heading_ix(query_seq)
            navi.append_data(qid)
            
            print(rid,navi_mp,ix_mp)
            print(yaw,navi_hx,ix_hx)
            qdata = [qid,rid,navi_mp,ix_mp,yaw,navi_hx,ix_hx]

            if abs(rid-navi_mp) >= 10: 
                msm_nxi.append(qdata)
            if abs(yaw-navi_hx) > 10:
                msm_nxh.append(qdata)
            if abs(rid-ix_mp) >= 10: 
                msm_uxi.append(qdata)
            if abs(yaw-ix_hx) > 10:
                msm_uxh.append(qdata)

            for dkey,dval in zip(datakeys,qdata):
                simdata[dkey].append(dval)

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

    if mk_plots:
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
    
    if save_data:
        save_simdata(simdata,route_id)

    return navi,route,simdata, (xh,uxh,nxh,xi,uxi,nxi)

# navi,route,simdata, (xh,uxh,nxh,xi,uxi,nxi) = test(route_id=0)


# TODO: run exps and save data in dataframe or whatever
def multi_test():
    mdata = []
    for ri in range(8):
        navi,route,simdata, xhi = test(route_id=ri,
                                       route_type='basic',
                                       navi_type='spm')
        mdata.append([simdata,route.route_dict,navi.route_data,xhi])
    return mdata
# mdata = multi_test()

import os
import numpy as np
from copy import deepcopy
from auxs import *
from tqdm import tqdm
from source.utils import *
# from source.imgproc import * 
# from source.routedatabase import Route,load_routes
# import source.seqnav as spm
import pdb
import matplotlib.pyplot as plt

# TODO: 
# 3d plot rmf for 3 qimgs
# random pos tests
# 

def test_navi(navi_type='spm',route_type='basic',route_id=0,print_data=True):
    navi,rx,rx_imgs,qx_imgs = load_navi_and_route(navi_type=navi_type,route_type=route_type,route_id=route_id)
    rxyo,qps = get_corr_xy_points(rx) # rxyo: rx,ry,ro; qps: qx,qy, qrxy_id,qrx,qry
    navi_data = np.zeros((qps.shape[0],4)) # heading, id pointer (mp), window blim, window flim
    correct_data = np.zeros((qps.shape[0],2)) # heading, id
    fv_data = np.zeros((qps.shape[0],3)) # quick_id, heading, mp id
    # mem_ids = navi.segment_memory()
    # for i in range(len(mem_ids)-1):
    #     mi,mx = mem_ids[i:i+2]
    #     print(mi,mx)
    #     ii = navi.ims[mi:mx] - navi.ims[mi]
    #     plot_imgs([navi.ims[mi],navi.ims[mx-1],np.sum(ii,axis=(0))],rows=3,cols=1)
    # import pdb; pdb.set_trace()
    for ei,qxim in enumerate(qx_imgs):
        # get heading, id pointer (= mem pointer), ref window data & route corr data
        heading = navi.get_heading(qxim)
        navi_data[ei] = [heading,navi.mem_pointer,navi.blimit,navi.flimit]
        rx_grid_id, rx_mindist_id = qps[ei][2:4].astype(int)
        correct_heading = rxyo[rx_grid_id][2] # route heading, training
        correct_data[ei] = [correct_heading,rx_grid_id]
        # alt
        alt_heading,alt_mp = 0,0
        if ei >= 2:
            qid,alt_heading,alt_mp = navi.get_pos_heading(qx_imgs[ei-2:ei+1])
            fv_data[ei] = [alt_heading,qid,alt_mp]
        if print_data:
            print(ei)
            print('headings => navi: {}, correct: {}, alt heading: {}'.format(heading,round(correct_heading,2),alt_heading))
            print('pointer id => navi: {}, grid ei: {}, rx min: {}, alt id: {}'.format(navi.mem_pointer,rx_grid_id,int(qps[ei][3]),alt_mp))
        # if np.abs(heading-correct_heading) > 5:
        if np.abs(navi.mem_pointer-rx_grid_id) > 10 or np.abs(navi.mem_pointer-rx_mindist_id) > 10:
            plot_imgs([rotate(heading,qxim),rx_imgs[rx_grid_id],rx_imgs[navi.mem_pointer],rx_imgs[int(qps[ei][3])]],rows=2,cols=2,subtitles=['query image: {} (rotated by heading = {})'.format(ei,heading),'route/grid image: {}, heading: {}'.format(rx_grid_id,round(correct_heading,2)),'navi pointer img: {}'.format(navi.mem_pointer),'min dist pointer img: {}'.format(int(qps[ei][3]))])
            pdb.set_trace()

test_navi(route_type='curve')
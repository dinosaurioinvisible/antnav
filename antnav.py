
import sys
import os
import numpy as np
from copy import deepcopy
from auxs import *
from tqdm import tqdm
from source.utils import *
from source.imgproc import * 
from source.routedatabase import Route,load_routes
import source.seqnav as spm
import pdb
import matplotlib.pyplot as plt

# 0) check load
# rx = load_routes(os.path.join(os.getcwd(),'routes'),[0])[0]
# pp_settings = {}
# pp_settings['shape'] = (360,80)
# pp_settings['blur'] = True
# pipeline = Pipeline(**pp_settings)
# rx_imgs = pipeline.apply(rx.get_imgs())

# 1) check images
# rns = np.random.randint(0,len(rx_imgs),size=4)
# plot_imgs([rx_imgs[rni] for rni in rns])

# 2) trivial check: location & rotation
def trivial_tests(imgs,id=-1):
    # try trivial 
    id = id if id >= 0 else np.random.randint(0,360)
    print('id: {}'.format(id))
    ims = np.array(imgs).astype(int)
    imx = imgs[id] 
    # non rotated to all
    ii = ims - imx
    idx = np.where(np.sum(ii,axis=(1,2))==0)
    print('idx: {}'.format(idx))
    # rotated to all
    rdeg = np.random.randint(0,360)
    print('rot degree: {}'.format(rdeg))
    # simultaneously
    imrx = rotate(rdeg,imx)
    idrx = np.where(np.sum(ims-imrx,axis=(1,2))==0)
    if len(idrx) > 0:
        print('idxr: {}'.format(idrx))
    else:
        rot_cases = np.zeros((360,2)) # for angle: id,val,rotval
        for ri in range(360):
            rdi = (rdeg+ri)%360
            # print(rdi)
            imr = rotate(rdi,imx)
            iir = ims - imr
            idr = np.where(np.sum(iir,axis=(1,2))==0)[0][0]
            # double check only shared active
            # iirv = (ims[idr]-imr) * np.where((ims[idr]-imr)!=0,1,0)
            # rot_cases[ri] = [idr,np.sum(iir[idr]),np.sum(iirv)]
            rot_cases[ri] = [idr,np.sum(iir[idr])]
        rmin = np.min(rot_cases[:,1])
        rmin_ids = np.where(rot_cases[:,1]==rmin)
        rmin_cases = rot_cases[rmin_ids]
        rmin_list = list(set(list(rmin_cases[:,0])))
        print(rmin_list)
    # pdb.set_trace()
# trivial_tests(rx_imgs)

# 3) trivial compare: location & rotation
def trivial_compare(route=0,random_pos=False):
    # load and preprocess
    gpath = os.path.join(os.getcwd(),'grid70')
    rx = load_routes(os.path.join(os.getcwd(),'routes'),[route],grid_path=gpath)[0]
    pp_settings = {}
    pp_settings['shape'] = (360,80)
    pp_settings['blur'] = True
    pipeline = Pipeline(**pp_settings)
    rx_imgs = pipeline.apply(rx.get_imgs())
    qx_imgs = deepcopy(rx_imgs) # for now
    rx_dict = rx.route_dict
    navi = spm.SequentialPerfectMemory(rx_imgs,'mae')
    # for loc recognition (doesn't work with SPM)
    if random_pos:
        rxs = np.random.randint(0,len(rx_imgs),size=11)
        qx_imgs = [rx_imgs[i] for i in rxs] #ids?
    # get heading
    for ei,imx in enumerate(qx_imgs[:-1]):
        # known from dict
        loc = ei if not random_pos else rxs[ei]
        print('\nimg index: {}'.format(loc))
        ri = np.random.randint(0,360)
        qim = rotate(ri,imx)
        print('random rot angle: {}'.format(ri))
        yaw,yaw2 = rx_dict['yaw'][loc:loc+2]
        # yaw2 = rx_dict['yaw'][loc+1]
        dh_dict = yaw2 - yaw
        print('hx_dict: {}, dx: {} -> {}'.format(yaw,dh_dict,yaw2))
        # navi
        hx_navi = navi.get_heading(qim)
        dh_navi = (ri + hx_navi)%360
        h_navi = (yaw+dh_navi)%360
        print('hx_navi: {}, dx: {} -> {}'.format(hx_navi,dh_navi,h_navi))
        # v2, (basic case) frame recog
        hx_alt = ri # implicit
        dh_alt = navi.get_heading_alt(qim)
        h_alt = (yaw+dh_alt)%360
        print('hx_alt: {}, dx: {} -> {}'.format(hx_alt,dh_alt,h_alt))
        if dh_navi != dh_alt and not random_pos:
            pdb.set_trace()
        # v3, query seq, processed indep (for noisy env?)
        if loc >= 2:
            qim_seq = rx_imgs[loc-2:loc+1]
            h_seq = navi.get_heading_seq(qim_seq)
        # v4, int sequence recog (mult array across 3d)
        # qimseq = rx_imgs[loc-2:loc+1]
        # qimseq[-1] = qim*1
        # v5, loop break
    # pdb.set_trace()
    return navi
# navi = trivial_compare(route=0,random_pos=True)

# route_id = 0
# gpath = os.path.join(os.getcwd(),'grid70')
# route = Route(os.path.join(os.getcwd(),'routes', f'route{route_id}'), route_id=route_id,
#               grid_path=gpath, max_dist=0.3)

# 4) test
def test_navi(route_id,mp=False,refw=False,gdist=0.2,mk_plots=False,random_pos=False,print_data=False):
    # load and filter
    gpath = os.path.join(os.getcwd(),'grid70')
    rx = load_routes(os.path.join(os.getcwd(),'routes'),[route_id],grid_path=gpath,max_dist=gdist)[0]
    pp_settings = {}
    pp_settings['shape'] = (360,80)
    pp_settings['blur'] = True
    pipeline = Pipeline(**pp_settings)
    # route, query data for navi
    rx_imgs = pipeline.apply(rx.get_imgs())
    qx_imgs = pipeline.apply(rx.get_qimgs())
    navi = spm.SequentialPerfectMemory(rx_imgs,'mae')
    navi.mp = 0 if mp else -1
    navi.ref_window = True if refw else False
    rx_dict = rx.route_dict
    rxy = rx.get_xycoords()
    qxy = rx.get_qxycoords()
    # for random positioning test
    if random_pos:
        rxs = np.random.randint(0,len(rx_imgs),size=11)
    # data to plot
    cxyos = []
    navi_data = np.zeros((len(qx_imgs),3)) # quick id, heading, heading id
    # get location and heading
    for ei in range(len(qx_imgs)):
    # for ei in tqdm(range(len(qx_imgs))):
        qimx = qx_imgs[ei]
        mp0 = navi.mp
        loc = ei if not random_pos else rxs[ei]
        hx = navi.get_heading(qimx)

        # current location and initial heading
        cx = rx_dict['qx'][loc]
        cy = rx_dict['qy'][loc]
        co0 = rx_dict['qyaw'][loc] # always zero
        # get aprox location, heading & heading loc id
        quick_id,heading,heading_id = navi.get_pos_heading(qimx)
        # with only 1 img, you can't adjust the heading really (2 min are needed for + or -)
        ho = heading * 1 # for now
        cxyo = (cx,cy,co0,ho)
        cxyos.append(cxyo)

        # other test fxs
        # navi.get_seq_location(qimx)
        # navi.get_rot_location(qimx)
        ix_heading = navi.get_intx_location(qimx)

        # guess from route
        gx = rx_dict['x'][heading_id]
        gy = rx_dict['y'][heading_id]
        go = rx_dict['yaw'][heading_id]
        gxyo = (gx,gy,go)

        # print out, plot
        if print_data:
            print('\nimg index: {}/{}'.format(loc,len(qx_imgs)))
            print('current (query) pos: x: {}, y: {}, h0: {}'.format(cx,cy,co0))
            print('quick id: {}, navi heading: {}, heading: {}, heading id: {}'.format(quick_id,hx,heading,heading_id))
            print('guessed pos: x: {}, y: {}, h: {}'.format(gx,gy,go))
        navi_data[ei] = [quick_id,heading,heading_id]
        if mk_plots:
            plot_routes(rxy,qxy,[cxyo],[gxyo],title='mp: {}, quick loc: {}, head loc: {}, yaw: {}, heading: {}'.format(mp0,quick_id,heading_id,round(go,3),ho))
            if (ei+1)%11 == 0:
                pdb.set_trace()

        # check fail
        if np.abs(go-ho) > 2:
            print("\n\n")
            print('img id: {}'.format(loc))
            print('heading: {}'.format(ho))
            print('route heading: {}'.format(go))
            print('navi get heading: {}'.format(hx))
            print('heading id: {}'.format(heading_id))
            # whats the correlative/correct route image?
            plot_imgs([qimx,navi.ims[heading_id]])
            pdb.set_trace()
        
    plot_routes(rxy,qxy,cxyos=cxyos)
    pdb.set_trace()
    return navi_data

data = test_navi(0,
                 mp=True,refw=True,
                 gdist=0.3,
                 mk_plots=False,
                 print_data=False
                 )


# problem 1
# query images are less than route imgs 
# they go forward to fast in comparison


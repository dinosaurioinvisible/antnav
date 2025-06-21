
import numpy as np
from auxs import *
from pdb import set_trace as pp
import matplotlib.pyplot as plt
import tonic
from mkdata import EventScramble

#TODO: a function for all this

navi, route, ims, qims, rxyo, qxys = load_navi_and_route(navi_type='seq2seq',
                                                         route_type='basic',
                                                         route_id = 0,
                                                         navi_corrs=False)

# difs 
iids1 = ims[:-1] - ims[1:]
# iib = np.abs(iidx)
# normalized sums
# iidsums = iids1.sum(axis=(1,2))/28800
# iibs = iib.sum(axis=(1,2))/28800
iids2 = iids1[:-1] - iids1[1:]
iids3 = iids2[:-1] - iids2[1:]
iids4 = iids3[:-1] - iids3[1:]

ii = 1
t0,tf = 25,75
iidx = [ims,iids1,iids2,iids3,iids4][ii]
# nf = 5                      # n frames combined 
# th = 25                     # 10%
# ii_mask = np.where(np.abs(iidx/nf)>th,1,0)
# iidx = iidx * ii_mask

ix_size = iidx.nonzero()[0].shape[0]

ix = np.rec.array(None, dtype=[('x',np.uint16),
                                ('y',np.uint16),
                                ('t',np.float32),
                                ('p',np.uint16)],
                                shape = ix_size)

ix['t'], ix['y'], ix['x'] = iidx.nonzero()
ix['t'] = ix['t'] + 0.5 /10000
ix['p'] = np.where(iidx[iidx.nonzero()]>0,1,0)


tx_tj = tonic.transforms.TimeJitter(std=1,
                                 clip_negative=True,
                                 sort_timestamps=True)
ixe = tx_tj(ix)

# only for short windows (=< 50)
# tx_sc = EventScramble()
# ixe = tx_sc(ixe)

# ftime = 1
# tx_dn = tonic.transforms.Denoise(filter_time=ftime)
# ixe = tx_dn(ixe)

window = 1
ev_count = 1000
sensor_size = (360,80,2)
tx_tf = tonic.transforms.ToFrame(sensor_size=sensor_size,time_window=window)
# tx_tf = tonic.transforms.ToFrame(sensor_size=sensor_size,event_count=ev_count)
ixf = tx_tf(ixe)
# ani = tonic.utils.plot_animation(ixf)


# temporal and structural dimensions of events


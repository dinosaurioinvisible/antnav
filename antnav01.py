
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from source.utils import *
import pdb


from source.routedatabase import Route,load_routes
rx = load_routes(os.getcwd(),[0])[0] 
raw_images = rx.get_imgs() # list: 600 ims, 150 x 720 

from source.utils import pre_process
# sets: shape, blur, edge_range
sets = {'blur': True}
pp_images = pre_process(raw_images,sets) # list
ims = np.array(pp_images).astype(int) # tensor

# from source.utils import pick_im_matcher
# corr, dot, rmse, mse, mae, entropy
# matcher = pick_im_matcher('mae')

# from source.navs.navs import Navigator
# from source.navs.seqnav import Seq2SeqPerfectMemory
# navi = Seq2SeqPerfectMemory(pp_images)

class Agent:
    def __init__(self,route,im_id=100,pp=True,checks=False):
        self.rx = route
        self.checks = checks
        self.random_init(im_id,pp)
        self.get_heading()
        # self.mk_working_memory()
        # self.get_location()

    def random_init(self,im_id,pp):
        # if given image
        self.imi = im_id if 20 < im_id < 580 else np.random.randint(20,580)
        # random pos
        self.x = rx.route_dict['x'][self.imi]
        self.y = rx.route_dict['y'][self.imi]
        # self.h = np.random.randint(0,360)
        self.h = 90
        # get images
        ims = rx.get_imgs()
        if pp:
            sets = {'blur': True}
            ims = pre_process(ims,sets)
        self.ims = np.array(ims).astype(int)
        # original, current, query & ref imgs
        self.oim = rx.route_dict['imgs'][self.imi].astype(int)
        self.im = self.ims[self.imi]
        self.qim = np.roll(self.im,self.h*2,axis=1)
        self.ref = self.ims[self.imi-10:self.imi+11]
        # just to check
        if self.checks:
            self.check_init(pp)

    def mk_memory(self):
        # memory of changes
        pass

    def mk_working_memory(self):
        pass

    def get_location(self):
        # at least in theory, you could be wherever
        # so there should be a very fast way to compare 
        # qim against the whole memory in ims
        plt.imshow(self.qim,cmap='gray',aspect='auto')
        plt.show()
        self.cx = self.ims[self.imi-5:self.imi+1]

    # where do i get the past query images?
    def get_heading(self,px=4,fx=0):
        # previous + current
        self.wm_ims = self.ims[self.imi-px:self.imi+1]
        self.wm_intx = np.sum(np.array([self.wm_ims[i+1] - self.wm_ims[i] for i in range(px-1)]),axis=0)
        if fx>0:
            self.fx_ims = self.ims[self.imi+1:self.imi+fx+1]
        degrees = range(*(0,360),1)
        # total_search_angle = round((360 - 0) / 1)
        print('correct angle: {}'.format(rx.route_dict['yaw'][self.imi]))
        best,angle = 0,None
        qx = np.zeros((self.ims[0].shape)).astype(int)
        qrim = qx*1
        for ri,rot in enumerate(degrees):
            rim = rotate(rot,self.qim) # inverted?
            rimx = self.wm_intx * rim
            if np.sum(np.abs(rimx)) > best:
                best = np.sum(np.abs(rimx))
                qrim = rim*1
                qx = rimx*1
                qk = self.qim - rim 
                angle = rot
                print(rot,np.sum(rimx))
        heading = self.h+angle
        heading = heading if heading < 360 else heading-360
        print('angle: {}, heading?: {}'.format(angle,heading))
        print(qk)
        imgs = [self.wm_ims[0],self.wm_ims[-2],self.wm_ims[-1],self.qim,self.wm_intx,qrim,qx,qk] #np.abs(qx)]
        self.plot_imgs(imgs,rows=2,cols=4)

    # animated wm vs pm
    def plot_response(self):
        pass
    
    # just to check
    def check_init(self,pp):
        rows = 1 if not pp else 2
        fig,axs = plt.subplots(rows,2,figsize=(8,8))
        fig.suptitle('image: {}, x: {}, y: {}, heading: {}'.format(self.imi,round(self.x,3),round(self.y,3),self.h))
        # current pm image
        axs[0,0].imshow(self.im,cmap='gray',aspect='auto')
        axs[0,0].set_title('pm current')
        # rolled
        axs[0,1].imshow(self.qim,cmap='gray',aspect='auto')
        axs[0,1].set_title('query img, rolled by {}'.format(self.h*2))
        if pp:
            # before pp blurring
            axs[1,0].imshow(self.oim,cmap='gray',aspect='auto')
            axs[1,0].set_title('before pre-processing')
            # same rolled
            axs[1,1].imshow(np.roll(self.oim,self.h*2,axis=1),cmap='gray',aspect='auto')
            axs[1,1].set_title('before pp rolled')
        plt.show()
        plt.close()

    def plot_imgs(self,imgs,cols=0,rows=0,title=''):
        rows = rows if rows > 0 else 1
        cols = cols if cols > 0 else len(imgs)
        fig,axs = plt.subplots(rows,cols,figsize=(16,8))
        fig.suptitle(title)
        for ei,ax in enumerate(axs.flat):
            im = ax.imshow(imgs[ei],vmin=-255,vmax=255,cmap='gray',aspect='auto')
        # cbar = fig.colorbar(im,location='bottom')
        plt.show()
        plt.close()
ax = Agent(rx)
# qxs = np.random.randint(20,550,size=10)
# for qi in qxs:
#     ax.imi = qi
#     ax.im = ax.ims[ax.imi]
#     ax.qim = np.roll(ax.im,ax.h*2,axis=1)
#     ax.get_heading()



def plot_intx_min(imgs,id):
    fig,axs = plt.subplots(1,3,figsize=(15,5))
    intx = imgs[id+1] - imgs[id]
    print(intx)
    data = [imgs[id],imgs[id+1],intx]
    for ei,ax in enumerate(axs.flat):
        im = ax.imshow(data[ei],vmin=-255,vmax=255,cmap='gray',aspect='auto')
    # cbar = fig.colorbar(im,location='bottom')
    plt.show()
    plt.close()
    return intx
# ix = plot_intx_min(ims,100)

def plot_intx_fixed_seq(imgs,id,window=5):
    data = [imgs[id+i] for i in range(window)]
    intx = np.sum(np.array([imgs[id+i+1] - imgs[id+i] for i in range(window-1)]),axis=0)
    print(intx)
    data.append(intx)
    # pdb.set_trace()
    fig,axs = plt.subplots(2,3,figsize=(15,8))
    for ei,ax, in enumerate(axs.flat):
        im = ax.imshow(data[ei],vmin=-255,vmax=255,cmap='gray',aspect='auto')
    cbar = fig.colorbar(im,location='bottom')
    plt.show()
    plt.close()
    return intx
# ix = plot_intx_fixed_seq(ims,130)

def intx_open_seq(imgs,id,thx=200,mk_plot=False):
    intx = np.zeros((imgs.shape[1:])).astype(int)
    id0 = id*1
    cx = True
    while cx == True:
        intx += imgs[id+1] - imgs[id]
        if np.max(np.abs(intx)) > thx:
            cx = False
        if id == imgs.shape[0]-2:
            print('\nend of images')
            cx = False
        id += 1
    wx_ids = np.array([id0,id])
    print('\nstart: {}, stop: {}, max val: {}'.format(id0,id,np.max(np.abs(intx))))
    if mk_plot:
        data = [imgs[id0],imgs[id],intx]
        fig,axs = plt.subplots(1,3,figsize=(15,5))
        for ei,ax in enumerate(axs.flat):
            im = ax.imshow(data[ei],vmin=-255,vmax=255,cmap='gray',aspect='auto')
            if ei < 2:
                ax.set_title('img {}'.format(wx_ids[ei]))
            else:
                ax.set_title('window: {} - {}, {} imgs'.format(id0,id,id-id0))
        cbar = fig.colorbar(im,location='bottom')
        plt.show()
        plt.close()
    return intx,wx_ids
# ix,ids = intx_open_seq(ims,100,mk_plot=True)

def mk_memory_seqs(imgs,thx=200,mk_plots=False):
    # negatives are new, positives are old
    id = 0
    ixs = []
    ids = []
    while id < imgs.shape[0]-1:
        ix,wx_ids = intx_open_seq(imgs,id,thx=thx,mk_plot=mk_plots)
        id = wx_ids[1]
        ixs.append(ix)
        ids.append(wx_ids)
        print('id0: {}, idx: {}, window size: {}'.format(wx_ids[0],id,id-wx_ids[0]))
    ixs = np.array(ixs)
    ids = np.array(ids)
    # print data
    wx_sizes = ids[:,1] - ids[:,0]
    wx_max,wx_min = np.max(wx_sizes),np.min(wx_sizes)
    max_ids = ids[np.where(wx_sizes==wx_max)][0]
    min_ids = ids[np.where(wx_sizes==wx_min)][0]
    print('\nnumber of windows: {}'.format(len(ixs)))
    print('largest window: {} imgs ({}-{}), shortest: {} imgs ({}-{})'.format(wx_max,max_ids[0],max_ids[1],wx_min,min_ids[0],min_ids[1]))
    return ixs,ids
# ixs,ids = mk_memory_seqs(ims,thx=180,mk_plots=False)

def plot_imgs_row(imgs,title=''):
    fig,axs = plt.subplots(1,len(imgs),figsize=(18,6))
    fig.suptitle(title)
    for ei,ax in enumerate(axs.flat):
        im = ax.imshow(imgs[ei],vmin=-255,vmax=255,cmap='gray',aspect='auto')
    cbar = fig.colorbar(im,location='bottom')
    plt.show()
    plt.close()

# 1. use intx as wm
# 2. use dx as basic intx maybe
def compare_query(imgs,ixs,ids,qxs=[]):
    qxs = qxs if len(qxs) > 10 else np.random.randint(20,590,size=10)
    cxs = 0
    print()
    for qi in qxs:
        qx = imgs[qi]
        id0 = qi-10
        intx = np.sum(np.array([imgs[id0+i+1] - imgs[id0+i] for i in range(11)]),axis=0)
        dxs = ixs * intx #* np.where(intx<0,-1,1)
        dxs_sums = np.sum(dxs,axis=(1,2))
        # negatives and positives are getting confused (- * - = +)
        dx_id = np.where(dxs_sums==np.max(dxs_sums))[0]
        dx = dxs[dx_id][0]
        # int images
        ix,ix_ids = ixs[dx_id][0], ids[dx_id][0]
        try:
            cx_id = np.where((ids-qi)[:,0]>=0)[0][0]-1
        except:
            cx_id = ids.shape[0]-1
        cx,cx_ids = ixs[cx_id],ids[cx_id]
        print('query id: {}, cx window: {} - {}, dx window: {} - {}'.format(qi,cx_ids[0],cx_ids[1],ix_ids[0],ix_ids[1]))
        plot_imgs_row([qx,cx,ix,intx,dx],title='qx (qi: {}), correct wx: ({}-{}), ix (wx ids: {}-{}), intx, dx'.format(qi,cx_ids[0],cx_ids[1],ix_ids[0],ix_ids[1]))
        if ix_ids[0] <= qi <= ix_ids[1]:
            cxs += 1
    print('{}/{} correct matchings'.format(cxs,len(qxs)))
# ixs,ids = mk_memory_seqs(ims,thx=200,mk_plots=False)
# compare_query(ims,ixs,ids)

def plot_anim_v1(imgs,start,stop):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
    id = start
    fs = stop - start - 1
    im = ax1.imshow(imgs[id],cmap='gray',aspect='auto',animated=True)
    intx = imgs[start+1] - imgs[start]
    ax2.imshow(intx,cmap='gray',aspect='auto')

    def update_fig(id):
        if id == stop:
            id = start
        id += 1
        im.set_array(imgs[id])
        return im,

    ani = animation.FuncAnimation(fig,update_fig,interval=500,blit=True,repeat=True)
    plt.show()










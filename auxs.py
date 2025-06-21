
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from source.routedatabase import load_routes
from source.imgproc import *
from source.navs import seqnav

def plot_img(img,mk_gray=False):
    if mk_gray:
        plt.imshow(img,cmap='gray',aspect='auto')
    else:
        plt.imshow(img,aspect='auto')
    plt.show()
    plt.close()

def plot_imgs(imgs,rows=0,cols=0,title='',mk_gray=False,mk_cbar=False,subtitles=[]):
    if rows==0 and cols==0:
        if len(imgs)==4:
            rows,cols = 2,2
        elif len(imgs)==6:
            rows,cols = 2,3
        elif len(imgs)==8:
            rows,cols = 4,4
    rows = rows if rows > 0 else 1
    cols = cols if cols > 0 else len(imgs)
    fig,axs = plt.subplots(rows,cols,figsize=(8,8))
    fig.suptitle(title)
    for ei,ax in enumerate(axs.flat):
        if mk_gray:
            im = ax.imshow(imgs[ei],cmap='gray',aspect='auto')
        else:
            im = ax.imshow(imgs[ei],aspect='auto')
        # im = ax.imshow(imgs[ei],vmin=-255,vmax=255,cmap='gray',aspect='auto')
        if len(subtitles) == len(imgs):
            ax.title.set_text(subtitles[ei])
    if mk_cbar:
        cbar = fig.colorbar(im,location='bottom')
    fig.tight_layout()
    plt.show()
    plt.close()

def animplot(iims, rows=0, cols=0, step=100, color='gray'):
    if not isinstance(iims, list):
        iims = [iims]
    nf = iims[0].shape[0]
    if rows + cols == 0:
        rows = 1 if len(iims) <= 2 else 2
        cols = 2 if len(iims) <= 2 else int(len(iims)/2) + len(iims)%2
    fig, axs = plt.subplots(rows,cols, figsize=(10,5))
    ti = 0
    ims = []
    fig.suptitle('')
    for ei,ax in enumerate(axs.flat):
        im = ax.imshow(iims[ei][ti], cmap=color, aspect='auto', animated=True)
        ims.append(im)
    def update_fig(ti):
        ti = (ti+1)%nf
        fig.suptitle(f'{ti}/{nf}')
        for ui,ax in enumerate(axs.flat):
            ims[ui].set_array(iims[ui][ti])
        return [im for im in ims]
    ani = animation.FuncAnimation(fig,update_fig,interval=step,blit=False,repeat=True,cache_frame_data=False)
    plt.show()
    plt.close()

# def plot_seq_vs_img(img_seq,img,step=500):
#     fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
#     ii = 0
#     im = ax1.imshow(img_seq[ii],cmap='gray',aspect='auto',animated=True)
#     ax2.imshow(img,cmap='grey',aspect='auto')
#     def update_fig(ii):
#         ii = (id+1)%len(img_seq)
#         im.set_array(img_seq[ii])
#         return im, 
#     ani = animation.FuncAnimation(fig,update_fig,interval=step,blit=True,repeat=True)
#     # def onClick(event):
#     #     import pdb; pdb.set_trace()
#     plt.show()
#     plt.close()

def plot_routes(rxy,qxy,cxyos=[],gxyos=[],title=''):
    # route and grid query imgs
    plt.scatter(rxy['x'], rxy['y'],color='blue')
    plt.scatter(qxy['x'], qxy['y'],color='orange')
    # plt.plot(qxy['x'], qxy['y'],color='gray')
    if len(cxyos) > 0:
        # current pos from query data + obtained heading
        for cxyo in cxyos:
            plt.scatter(cxyo[0],cxyo[1],s=200,color='black')
            # default heading (always zero it seems)
            qo = 90 - cxyo[2]
            qo = np.radians(qo)
            arrqx = 2 * np.cos(qo)
            arrqy = 2 * np.sin(qo)
            plt.arrow(cxyo[0],cxyo[1],arrqx,arrqy,ec='gray')
            # heading
            ho = 90 - cxyo[3]
            ho = np.radians(ho)
            arrhx = 2.5 * np.cos(ho)
            arrhy = 2.5 * np.sin(ho)
            plt.arrow(cxyo[0],cxyo[1],arrhx,arrhy,ec='black')
    if len(gxyos) > 0:
        # guessed 
        for gxyo in gxyos:
            plt.scatter(gxyo[0],gxyo[1],s=100,color='red')
            go = 90 - gxyo[2]
            go = np.radians(go)
            arrgx = 2 * np.cos(go)
            arrgy = 2 * np.sin(go)
            plt.arrow(gxyo[0],gxyo[1],arrgx,arrgy,ec='red')
    if len(title) > 0:
        plt.suptitle(title)
    plt.show()
    plt.close()

# quick load of all the basics
def load_navi_and_route(navi_type='spm',route_type='',route_id=0,
                        navi_corrs=True,queue_size=3,sub_window=9):
    dirpath = os.path.abspath(os.path.join(os.getcwd(),'..','antworld_data'))
    if route_type == 'basic':
        rx_path = os.path.join(dirpath,'routes')
    elif route_type == 'set1':
        rx_path = os.path.join(dirpath,'set1','exp1','routes')
    elif route_type == 'curve':
        rx_path = os.path.join(dirpath,'curve-bins')
    gpath = os.path.join(dirpath,'grid70')
    route = load_routes(rx_path,[route_id],grid_path=gpath)[0]
    pp_settings = {}
    pp_settings['shape'] = (360,80)
    pp_settings['blur'] = True
    pipeline = Pipeline(**pp_settings)
    rx_imgs = pipeline.apply(route.get_imgs())
    qx_imgs = pipeline.apply(route.get_qimgs())
    if navi_type == 'spm':
        navi = seqnav.SequentialPerfectMemory(rx_imgs,'mae')
    elif navi_type == 'seq2seq':
        navi = seqnav.Seq2SeqPerfectMemory(rx_imgs,matcher='mae',
                                           queue_size=queue_size,
                                           sub_window=sub_window)
    if navi_corrs:
        navi.mk_corr_qr_points(route)
        return navi,route,np.array(rx_imgs),np.array(qx_imgs)
    else:
        rxyo, qxys = get_corr_xy_points(route)
        return navi,route,np.array(rx_imgs),np.array(qx_imgs),rxyo,qxys
    

# save 
def save_simdata(simdata,route_id):
    import pandas as pd
    dirpath = os.path.abspath(os.path.join(os.getcwd(),'..','antworld_data'))
    data = pd.DataFrame(simdata)
    data.to_csv(os.path.join(dirpath,'simdata_r{}.csv'.format(route_id)), index=False)

# now in seqnav class
def get_corr_xy_points(route):
    rcs = route.get_xycoords()
    qcs = route.get_qxycoords()
    # rxyo: rx, ry, ro
    rxyo = np.zeros((rcs['x'].shape[0],3))
    rxyo[:,0] = rcs['x']
    rxyo[:,1] = rcs['y']
    rxyo[:,2] = route.get_yaw()
    # qxys: qx, qy, qxy_corr_route_id
    qxys = np.zeros((qcs['x'].shape[0],3))
    qxys[:,0] = qcs['x']
    qxys[:,1] = qcs['y']
    # route id for qx,qy 
    for qi in range(qxys.shape[0]):
        qr_id,qr_dist,(qrx,qry) = route.min_dist_from_route(qxys[qi][:2])
        if qrx == rxyo[qr_id][0] and qry == rxyo[qr_id][1]:
            qxys[qi][2] = qr_id
        else:
            import pdb; pdb.set_trace() # just in case (never happened so far)
    return rxyo,qxys



# 
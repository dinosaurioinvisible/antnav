from source.utils import pick_im_matcher, dot_dist, mae, rmse, cor_dist, rmf, seq2seqrmf, pair_rmf, cos_sim, mean_angle, rotate
from source.analysis import d2i_rmfs_eval
import numpy as np
import copy
from collections import deque
from source.imgproc import Pipeline

class SequentialPerfectMemory:

    def __init__(self, route_images, matching, deg_range=(-180, 180), degree_shift=1, 
                window=20, dynamic_range=0.1, w_thresh=None, mid_update=True, sma_size=3, 
                **kwargs):
        self.route_end = len(route_images)
        self.route_images = route_images
        self.deg_range = deg_range
        self.deg_step = degree_shift
        self.degrees = np.arange(*deg_range)
        self.matcher = pick_im_matcher(matching)
        # if the dot product distance is used we need to make sure the images are standardized
        if self.matcher == dot_dist:
            self.pipe = Pipeline(normstd=True)
            self.route_images = self.pipe.apply(route_images)
        else: 
            self.pipe = Pipeline()

        # Log Variables
        self.recovered_heading = []
        self.logs = []
        self.window_log = []
        self.matched_index_log = []
        self.confidence = [1] * self.route_end
        self.window_sims = []
        self.best_sims = []
        self.window_headings = []
        self.CMA = []
        self.sma_qmet_log = []
        # append a starting value for the d2i qiality metric log
        # TODO: the metrics shouls proapblly be classes that each have their own
        # initialisation values etc
        self.sma_qmet_log.append(0)
        # Matching variables

        self.argminmax = np.argmin
        self.prev_match = 0.0

        # Window parameters
        self.starting_window = abs(window)
        if window < 0:
            self.window = abs(window)
            self.adaptive = True
            self.upper = int(round(self.window/2))
            self.lower = self.window - self.upper
            self.mem_pointer = self.window - self.upper
            self.w_thresh = w_thresh
            if sma_size:
                self.sma_size = sma_size
                #self.idf_sma = []
        else:
            self.window = window
            self.adaptive = False
            self.mem_pointer = 0
            self.upper = window
            self.lower = 0
        self.blimit = 0
        self.flimit = self.window

        # Adaptive window parameters
        self.dynamic_range = dynamic_range
        self.min_window = 10
        self.window_margin = 5
        self.deg_diff = 5
        self.agreement_thresh = 0.9

        # heading parameters
        self.qmet_q = deque(maxlen=3)

        # for ix fxs
        self.idx = 0    # mem pointer
        self.hix = 0    # heading
        self.window_blim = 0
        self.window_flim = 0
        self.route_data = dict.fromkeys(['x','y','heading','mp'],np.array([]))
    
    #TODO Need a better name for this function
    def reset_window(self, pointer):
        '''
        Resets the memory pointer assuming and the window size
        '''
        self.mem_pointer = pointer
        #self.window =self.starting_window

        # update upper an lower margins
        self.upper = int(round(self.window/2))
        self.lower = self.window - self.upper

        # Update the bounds of the window
        # the window limits bounce back near the ends of the route
        self.blimit = max(0, self.mem_pointer - self.lower)
        self.flimit = min(self.route_end, self.mem_pointer + self.upper)

    # get correlated query to route points, for testing
    def mk_corr_qr_points(self, route):
        # rxyo: rx, ry, r yaw
        self.rxyo = np.zeros((len(self.route_images),3))
        self.rxyo[:,0] = route.route_dict['x']
        self.rxyo[:,1] = route.route_dict['y']
        self.rxyo[:,2] = route.route_dict['yaw']
        # qxys: qx, qy, query correlative route id
        self.qxys = np.zeros((len(route.route_dict['qimgs']),3))
        self.qxys[:,0]  = route.route_dict['qx']
        self.qxys[:,1]  = route.route_dict['qy']
        for qi in range(self.qxys.shape[0]):
            qr_id, qr_dist, (qrx,qry) = route.min_dist_from_route(self.qxys[qi][:2])
            self.qxys[qi][2] = qr_id

    def append_data(self,qx_id):
        qx,qy = self.qxys[qx_id][:2]
        self.route_data['x'] = np.append(self.route_data['x'],qx)
        self.route_data['y'] = np.append(self.route_data['y'],qy)
        self.route_data['heading'] = np.append(self.route_data['heading'],self.hix)
        self.route_data['mp'] = np.append(self.route_data['mp'],self.idx)

    # if global, kind of rdf over the whole img set, for testing
    # basically a quick aprox when without memory window
    def get_aprox_location(self, query_img, bl=0, fl=0):
        rot_vals,rot_ids = np.zeros(360),np.zeros(360)
        ims = self.route_images if bl+fl==0 else self.route_images[bl:fl]
        for di in range(360):
            ir_difs = ims - rotate(di,query_img)
            ir_sums = np.sum(np.abs(ir_difs),axis=(1,2))
            ir_id = np.where(ir_sums==np.min(ir_sums))[0][0]    # may be more than one, very rare
            rot_vals[di] = np.min(ir_sums)
            rot_ids[di] = ir_id
        heading = int(np.where(rot_vals==np.min(rot_vals))[0])  # min -> heading
        heading_id = bl + int(rot_ids[heading])                 # best heading -> id
        return heading, heading_id
    
    # update mem window using query img + mem pointer
    def mk_mem_window(self,qim):
        flw = self.window_flim + 10
        blw = self.window_blim
        hx, hx_id = self.get_aprox_location(qim,bl=blw,fl=flw)
        # print(hx_id,self.idx)
        idb,idf = min(hx_id,self.idx), max(hx_id,self.idx)
        # idf will be almost always very near the center
        self.window_flim = min(idf+11,len(self.route_images)-1) if idb > 0 else idf + self.window
        self.window_blim = max(0,self.window_flim - self.window)        
        # return self.window_blim,self.window_flim
    
    # get id using a sequence of query images
    def get_heading_ix(self, query_seq):
        # get aprox window from query img
        query_img = query_seq[-1]
        self.mk_mem_window(query_img)
        # self.window_blim,self.window_flim = self.mk_mem_window(query_img)
        
        # get id and heading
        if type(query_seq) == list:
            query_seq = np.array(query_seq)
        wsize = query_seq.shape[0]        # sliding tensor window size = query seq
        subw_lims, qr_difs, qr_sums = [],[],[]
        window_flim = min(self.window_flim-wsize,len(self.route_images)-wsize)
        for wi in range(self.window_blim,window_flim):
            rx_ims = self.route_images[wi:wi+wsize]           # route mem img seq
            qr_dif = rx_ims - query_seq                       # mem qx difference
            subw_lims.append((wi,wi+wsize))
            qr_difs.append(qr_dif)
            qr_sums.append(np.sum(qr_dif))

        qr_min = np.min(np.abs(qr_sums))                            # minima (intensity wise)
        qr_id = int(np.where(np.abs(qr_sums)==qr_min)[0])           # min id
        subw_blim, subw_flim = subw_lims[qr_id]                     # sub window limits
        subw_sums = np.abs(np.sum(qr_difs[qr_id],axis=(1,2)))       # kind of recurs, to get id within subw
        subw_idx = int(np.where(subw_sums==np.min(subw_sums))[0])   # min within sub window
        self.idx = subw_blim + subw_idx                             # update mem pointer ix

        # subw_absums = np.sum(np.abs(qr_difs[qr_id]),axis=(1,2))     # subw sums for heading
        # hx_id = self.argminmax(subw_absums)                         # min 
        # self.hix = self.get_heading_ix(query_seq[hx_id],self.route_images[max(0,subw_blim-wsize):subw_flim+wsize])
        
        wrsims = seq2seqrmf(query_seq, self.route_images[max(0,subw_blim-wsize):subw_flim+wsize])
        wind_sims = []
        wind_headings = []
        indices = self.argminmax(wrsims, axis=1)
        for i, idi in enumerate(indices):
            wind_sims.append(wrsims[i, idi])
            wind_headings.append(self.degrees[idi])
        heading_id = int(round(self.argminmax(wind_sims)))
        self.hix = wind_headings[heading_id] % 360

        return self.idx,self.hix

    # TODO: this is really difficult, nothings seems to capture invariance well
    def segment_memory(self):
        ims = np.array(self.route_images)
        im_txs = ims[:-1] - ims[1:]                         # txs diffs
        iix = ims * 0 + 1
        for i in range(1,ims.shape[0]):
            dxi = np.abs(ims[i] - ims[i-1])                 # only if dif > previous dif
            iix[i] = np.where(dxi > iix[i-1], dxi, 0)       # min form of attention
        iix = iix[1:]                                       # only txs
        ux = iix/255                                        # norm, sort of weights also
        imxs = np.where(np.abs(im_txs) * iix > ux, im_txs, 0)
        return imxs

    # TODO: quick clean for difficult obstructions
    def remk_obstructions(self,threshold=1000):
        ims = np.array(self.route_images)
        iix = ims[:-1] - ims[1:]
        osums = np.sum(np.abs(iix),axis=(1,2))/1000
        ob_txs = np.where(osums>threshold)[0]
        ob_seqs = []
        for i in range(osums.shape[0]-3):
            if np.sum(osums[i:i+3]) > threshold * 3:
                ob_seqs.append([i,i+3])
        ob_seqs = np.array(ob_seqs)


    def get_heading(self, query_img):
        '''
        Recover the heading given a query image
        :param query_img:
        :return:
        '''
        # commented these, because query images are already preprocessed in my case
        # query_img = self.pipe.apply(query_img)
        # 
        # get the rotational similarities between a query image and a window of route images
        wrsims = rmf(query_img, self.route_images[self.blimit:self.flimit], self.matcher, self.deg_range, self.deg_step)
        self.window_log.append([self.blimit, self.flimit])
        # Holds the best rot. match between the query image and route images
        wind_sims = []
        # Recovered headings for the current image
        wind_headings = []
        # get best similarity match adn index w.r.t degrees
        indices = self.argminmax(wrsims, axis=1)
        for i, idx in enumerate(indices):
            wind_sims.append(wrsims[i, idx])
            wind_headings.append(self.degrees[idx])

        # Save the best degree and sim for window similarities
        self.window_sims.append(wind_sims)
        self.window_headings.append(wind_headings)
        # append the rsims of all window route images for that query image
        self.logs.append(wrsims)
        # find best image match and heading
        idx = int(round(self.argminmax(wind_sims)))
        self.best_sims.append(wind_sims[idx])
        heading = wind_headings[idx]
        self.recovered_heading.append(heading)

        # log the memory pointer before the update
        # mem_pointer - upper can cause the calc_dists() to go out of bounds
        matched_idx = self.blimit + idx
        self.matched_index_log.append(matched_idx)

        #evaluate ridf
        # h_eval = self.eval_ridf(wrsims[idx])

        if self.adaptive:
            best = wind_sims[idx]
            # TODO here I need to make the updating function modular
            self.dynamic_window_log_rate(best)
            self.check_w_size()

        # Update memory pointer
        self.update_mid_pointer(idx)
        return heading

    def eval_ridf(self, ridf):
        '''
        Evaluates the ridf quality
        returs: True if quality is good False if quality is bad
        '''
        quality = d2i_rmfs_eval(ridf).item()
        self.qmet_q.append(quality)
        
        sma = sum(self.qmet_q) / len(self.qmet_q)
        
        if sma < self.sma_qmet_log[-1]:
            self.sma_qmet_log.append(sma)
            return False
        self.sma_qmet_log.append(sma)
        return True

    def update_pointer(self, idx):
        '''
        Update the mem pointer to the back of the window
        mem_pointer = blimit
        :param idx:
        :return:
        '''
        self.mem_pointer += idx
        # in this case the upperpart is equal to the upper margin
        self.upper = self.window
        # Update the bounds of the window
        # the window limits bounce back near the ends of the route
        self.blimit = max(0, self.mem_pointer)
        self.flimit = min(self.route_end, self.mem_pointer + self.upper)

    def update_mid_pointer(self, idx):
        '''
        Update the mem pointer to the middle of the window
        :param idx:
        :return:
        '''
        # Update memory pointer
        self.mem_pointer = self.blimit + idx

        # update upper an lower margins
        self.upper = int(round(self.window/2))
        self.lower = self.window - self.upper

        # Update the bounds of the window
        # the window limits bounce back near the ends of the route
        self.blimit = max(0, self.mem_pointer - self.lower)
        self.flimit = min(self.route_end, self.mem_pointer + self.upper)
        

    def check_w_size(self):
        self.window = self.route_end if self.window > self.route_end else self.window

    def get_agreement(self, window_headings):
        a = np.full(len(window_headings), 1)
        return cos_sim(a, window_headings)

    def consensus_heading(self, wind_headings, h):
        '''
        Calculates the agreement of the window headings.
        If the agreement is above the threshold the best heading is used
        otherwise the last heading is used.
        :param wind_headings:
        :param h:
        :return:
        '''
        if self.get_agreement(wind_headings) >= self.agreement_thresh:
            self.recovered_heading.append(h)
        elif len(self.recovered_heading) > 0:
            self.recovered_heading.append(self.recovered_heading[-1])
        else:
            self.recovered_heading.append(h)

    def evaluated_heading(self, ridf_eval):        
        if ridf_eval: # if quality is good
            return self.recovered_heading[-1]
        else: #if qiality is bad
            self.recovered_heading[-1] = 0
            return self.recovered_heading[-1]
            

    def average_heading2(self, h):
        '''
        Calculates the average of the last window heading and the current window heading
        :param h:
        :return:
        '''
        if len(self.recovered_heading) > 0:
            self.recovered_heading.append(mean_angle([h, self.recovered_heading[-1]]))
        else:
            self.recovered_heading.append(h)

    def average_headings(self, wind_heading):
        '''
        Calculates the average of all window headings
        :param wind_heading:
        :return:
        '''
        self.recovered_heading.append(mean_angle(wind_heading))

    def dynamic_window_con(self, best):
        '''
        Change the window size by a constant depending on the best img match gradient.
        If the last best img sim > the current best img sim the window grows
        and vice versa
        :param best:
        :return:
        '''
        # Dynamic window adaptation based on match gradient.
        if best > self.prev_match or self.window <= self.min_window:
            self.window += self.window_margin
        else:
            self.window -= self.window_margin
        self.prev_match = best
    
    def dynamic_window_rate(self, best):
        '''
        Change the window size depending on the current best and previous img match gradient. 
        Update the size by the dynamic_rate (percetage of the window size)
        :param best:
        :return:
        '''
        # Dynamic window adaptation based on match gradient.
        if best > self.prev_match or self.window <= self.min_window:
            self.window += round(self.window * self.dynamic_range)
        else:
            self.window -= round(self.window * self.dynamic_range)
        self.prev_match = best

    def dynamic_window_log_rate(self, best):
        '''
        Change the window size depending on the current best and previous img match gradient. 
        Update the size by log of the current window size
        :param best:
        :return:
        '''
        # Dynamic window adaptation based on match gradient.
        if best > self.prev_match or self.window <= self.min_window:
            self.window += round(self.min_window/np.log(self.window))
        else:
            self.window -= round(np.log(self.window))
        self.prev_match = best
    
    def dynamic_window_sma_log_rate(self, best):
        '''
        Change the window size depending on the current best and SMA of past mathes gradient. 
        Update the size by log of the current window size
        :param best:
        :return:
        '''
        # Dynamic window adaptation based on SMA match gradient.
        idfmin_sma = np.mean(self.best_sims[max(-self.sma_size, -len(self.best_sims)):])
        if best > idfmin_sma or self.window <= self.min_window:
            self.window += round(self.min_window/np.log(self.window))
        else:
            self.window -= round(np.log(self.window))
    
    def thresh_dynamic_window_log_rate(self, best):
        '''
        Change the window size depending on the current best and previous img match gradient. 
        Update the size by log of the current window size
        :param best:
        :return:
        '''
        # Dynamic window adaptation based on match gradient thresh.
        perc_cng = (best - self.prev_match + np.finfo(np.float).eps)/(self.prev_match + np.finfo(np.float).eps)
        if perc_cng > self.w_thresh or self.window <= self.min_window:
            self.window += round(self.min_window/np.log(self.window))
        elif perc_cng < -self.w_thresh:
            self.window -= round(np.log(self.window))
        else: pass
        self.prev_match = best

    def dynamic_window_h2(self, h):
        '''
        Change the window size depending on the best heading gradient.
        If the difference between the last heading and the current heading is > self.deg_diff
        then the window grows and vice versa
        :param h:
        :return:
        '''
        diff = abs(h - self.recovered_heading[-1])
        if diff > self.deg_diff or self.window <= self.min_window:
            self.window += self.window_margin
        else:
            self.window -= self.window_margin

    def dynamic_window_h(self, wind_headings):
        '''
        The window grows if the window headings disagree and vice versa
        :param wind_headings:
        :return:
        '''
        if self.get_agreement(wind_headings) <= self.agreement_thresh or self.window <= self.min_window:
            self.window += self.window_margin
        else:
            self.window -= self.window_margin


    def navigate(self, query_imgs):
        assert isinstance(query_imgs, list)
        for query_img in query_imgs:
            self.get_heading(query_img)
        return self.recovered_heading, self.window_log
    

    def get_rec_headings(self):
        return self.recovered_heading

    def get_index_log(self):
        return self.matched_index_log

    def get_window_log(self):
        return self.window_log

    def get_rsims_log(self):
        return self.logs

    def get_confidence(self):
        return self.confidence

    def get_window_sims(self):
        return self.window_sims

    def get_best_sims(self):
        return self.best_sims

    def get_window_headings(self):
        return self.window_headings

    def get_CMA(self):
        return self.CMA
    
    def get_name(self):
        if self.adaptive:
            return f'A-SMW({self.starting_window})'
        else:
            return f'SMW({self.window})'


class Seq2SeqPerfectMemory:
    
    def __init__(self, route_images, matching, deg_range=(-180, 180), degree_shift=1, window=20, dynamic_range=0.1, queue_size=3):
        self.route_end = len(route_images)
        self.route_images = route_images
        self.deg_range = deg_range
        self.deg_step = degree_shift
        self.degrees = np.arange(*deg_range)
        self.queue_size = queue_size
        self.queue = deque(maxlen=queue_size)
        self.matcher = pick_im_matcher(matching)
        # if the dot product distance is used we need to make sure the images are standardized
        if self.matcher == dot_dist:
            self.pipe = Pipeline(normstd=True)
            self.route_images = self.pipe.apply(route_images)
        else: 
            self.pipe = Pipeline()

        # for ix fxs
        self.idx = 0    # mem pointer
        self.hix = 0    # heading
        self.window_blim = 0
        self.window_flim = 0
        self.route_data = dict.fromkeys(['x','y','heading','mp'],np.array([]))

        # Log Variables
        self.recovered_heading = []
        self.logs = []
        self.window_log = []
        self.matched_index_log = []
        self.confidence = [1] * self.route_end
        self.window_sims = []
        self.best_sims = []
        self.window_headings = []
        self.CMA = []
        # Matching variables
        self.argminmax = np.argmin
        self.prev_match = 0.0

        # Window parameters
        if window < 0:
            self.window = abs(window)
            self.adaptive = True
            self.upper = int(round(self.window/2))
            self.lower = self.window - self.upper
            self.mem_pointer = self.window - self.upper
        else:
            self.window = window
            self.adaptive = False
            self.mem_pointer = 0
            self.upper = window
            self.lower = 0
        self.blimit = 0
        self.flimit = self.window

        # Adaptive window parameters
        self.dynamic_range = dynamic_range
        self.min_window = 10
        self.window_margin = 5
        self.deg_diff = 5
        self.agreement_thresh = 0.9

    def reset_window(self, pointer):
        self.mem_pointer = pointer
        self.flimit = self.mem_pointer + self.upper
        self.blimit = self.mem_pointer - self.lower
        # self.window_blim = self.blimit # to avoid bias on the end of routes

        if self.flimit > self.route_end:
            self.mem_pointer = (self.route_end - self.window) + self.lower
            self.flimit = self.route_end
            self.blimit = self.route_end - self.window
        if self.blimit <= 0:
            self.mem_pointer = self.lower
            self.blimit = 0
            self.flimit = self.mem_pointer + self.window
            # self.window_blim = self.blimit # same

    # get correlated query to route points, for testing
    def mk_corr_qr_points(self, route):
        # rxyo: rx, ry, r yaw
        self.rxyo = np.zeros((len(self.route_images),3))
        self.rxyo[:,0] = route.route_dict['x']
        self.rxyo[:,1] = route.route_dict['y']
        self.rxyo[:,2] = route.route_dict['yaw']
        # qxys: qx, qy, query correlative route id
        self.qxys = np.zeros((len(route.route_dict['qimgs']),3))
        self.qxys[:,0]  = route.route_dict['qx']
        self.qxys[:,1]  = route.route_dict['qy']
        for qi in range(self.qxys.shape[0]):
            qr_id, qr_dist, (qrx,qry) = route.min_dist_from_route(self.qxys[qi][:2])
            self.qxys[qi][2] = qr_id

    def append_data(self,qx_id):
        qx,qy = self.qxys[qx_id][:2]
        self.route_data['x'] = np.append(self.route_data['x'],qx)
        self.route_data['y'] = np.append(self.route_data['y'],qy)
        self.route_data['heading'] = np.append(self.route_data['heading'],self.hix)
        self.route_data['mp'] = np.append(self.route_data['mp'],self.idx)

    # if global, kind of rdf over the whole img set, for testing
    # basically a quick aprox when without memory window
    def get_aprox_location(self, query_img, bl=0, fl=0):
        rot_vals,rot_ids = np.zeros(360),np.zeros(360)
        ims = self.route_images if bl+fl==0 else self.route_images[bl:fl]
        for di in range(360):
            ir_difs = ims - rotate(di,query_img)
            ir_sums = np.sum(np.abs(ir_difs),axis=(1,2))
            ir_id = np.where(ir_sums==np.min(ir_sums))[0][0]    # may be more than one, very rare
            rot_vals[di] = np.min(ir_sums)
            rot_ids[di] = ir_id
        heading = int(np.where(rot_vals==np.min(rot_vals))[0])  # min -> heading
        heading_id = bl + int(rot_ids[heading])                 # best heading -> id
        return heading, heading_id
    
    # get id and heading using a sequence of query images
    def get_heading_ix(self, query_seq):
        # get aprox window from query img
        query_img = query_seq[-1]
        self.window_blim,self.window_flim = self.mk_mem_window(query_img)
        
        # get id and heading
        if type(query_seq) == list:
            query_seq = np.array(query_seq)
        wsize = query_seq.shape[0]        # sliding tensor window size = query seq
        subw_lims = []
        qr_difs = []
        qr_sums = []        
        # window_flim = min(self.flimit-wsize,len(self.route_images)-wsize)
        window_flim = min(self.window_flim-wsize,len(self.route_images)-wsize)
        for wi in range(self.window_blim,window_flim):
            rx_ims = self.route_images[wi:wi+wsize]                             # route mem img seq
            qr_dif = rx_ims - query_seq                                         # mem qx difference
            subw_lims.append((wi,wi+wsize))
            qr_difs.append(qr_dif)
            qr_sums.append(np.sum(qr_dif))

        qr_min = np.min(np.abs(qr_sums))                            # minima (intensity wise)
        qr_id = int(np.where(np.abs(qr_sums)==qr_min)[0])           # min id
        subw_blim, subw_flim = subw_lims[qr_id]                     # sub window limits
        subw_sums = np.abs(np.sum(qr_difs[qr_id],axis=(1,2)))       # kind of recurs, to get id within subw
        subw_idx = int(np.where(subw_sums==np.min(subw_sums))[0])   # min within sub window
        self.idx = subw_blim + subw_idx                            # update mem pointer ix

        # subw_absums = np.sum(np.abs(qr_difs[qr_id]),axis=(1,2))     # subw sums for heading
        # hx_id = self.argminmax(subw_absums)                         # min 
        # self.hix = self.get_heading_ix(query_seq[hx_id],self.route_images[max(0,subw_blim-wsize):subw_flim+wsize])
        
        wrsims = seq2seqrmf(query_seq, self.route_images[max(0,subw_blim-wsize):subw_flim+wsize])
        wind_sims = []
        wind_headings = []
        indices = self.argminmax(wrsims, axis=1)
        for i, idi in enumerate(indices):
            wind_sims.append(wrsims[i, idi])
            wind_headings.append(self.degrees[idi])
        heading_id = int(round(self.argminmax(wind_sims)))
        self.hix = wind_headings[heading_id]%360
        
        return self.idx,self.hix
    
    # update mem window using query img + mem pointer
    def mk_mem_window(self,qim):
        flw = self.window_flim + 10
        blw = self.window_blim
        hx, hx_id = self.get_aprox_location(qim,bl=blw,fl=flw)
        # print(hx_id,self.idx)
        idb,idf = min(hx_id,self.idx), max(hx_id,self.idx)
        # idf will be almost always very near the center
        self.window_flim = min(idf+11,len(self.route_images)-1) if idb > 0 else idf + self.window
        self.window_blim = max(0,self.window_flim - self.window)
        
        return self.window_blim,self.window_flim
    
    # TODO: this is really difficult, nothings seems to capture invariance well
    def segment_memory(self):
        ims = np.array(self.route_images)
        im_txs = ims[:-1] - ims[1:]                         # txs diffs
        iix = ims * 0 + 1
        for i in range(1,ims.shape[0]):
            dxi = np.abs(ims[i] - ims[i-1])                 # only if dif > previous dif
            iix[i] = np.where(dxi > iix[i-1], dxi, 0)       # min form of attention
        iix = iix[1:]                                       # only txs
        ux = iix/255                                        # norm, sort of weights also
        imxs = np.where(np.abs(im_txs) * iix > ux, im_txs, 0)
        return imxs

    # TODO: quick clean for difficult obstructions
    def remk_obstructions(self,threshold=1000):
        ims = np.array(self.route_images)
        iix = ims[:-1] - ims[1:]
        osums = np.sums(np.abs(iix),axis=(1,2))/1000
        ob_txs = np.where(osums>threshold)[0]
        ob_seqs = []
        for i in range(osums.shape[0]-3):
            if np.sum(osums[i:i+3]) > threshold * 3:
                ob_seqs.append([i,i+3])
        ob_seqs = np.array(ob_seqs)


    def get_heading(self, query_img):
        '''
        Recover the heading given a query image
        :param query_img:
        :return:
        '''
        #If the query images queue is full then remove the oldest element 
        # and add the new image (removal happes automaticaly when using the maxlen argument for the deque)
        self.queue.append(query_img)
        # get the rotational similarities between the query images and a window of route images
        wrsims = seq2seqrmf(self.queue, self.route_images[self.blimit:self.flimit], self.matcher, self.deg_range, self.deg_step)
        
        self.window_log.append([self.blimit, self.flimit])
        # Holds the best rot. match between the query image and route images
        wind_sims = []
        # Recovered headings for the current image
        wind_headings = []
        # get best similarity match adn index w.r.t degrees
        indices = self.argminmax(wrsims, axis=1)
        for i, idx in enumerate(indices):
            wind_sims.append(wrsims[i, idx])
            wind_headings.append(self.degrees[idx])

        # Save the best degree and sim for window similarities
        self.window_sims.append(wind_sims)
        self.window_headings.append(wind_headings)
        # append the rsims of all window route images for that query image
        self.logs.append(wrsims)
        # find best image match and heading
        # the index needs to be modulo the size of the window 
        # because now the window sims are the size of current queque * window 
        idx = int(self.argminmax(wind_sims) % self.window)
        self.best_sims.append(wind_sims[idx])
        heading = wind_headings[idx]%360
        self.recovered_heading.append(heading)

        # log the memory pointer before the update
        # mem_pointer - upper can cause the calc_dists() to go out of bounds
        #!!! For the mid update only
        matched_idx = self.mem_pointer + (idx - self.lower)
        self.matched_index_log.append(matched_idx)

        if self.adaptive:
            best = wind_sims[idx]
            self.dynamic_window_log_rate(best)
            self.check_w_size()

        # Update memory pointer
        self.update_mid_pointer(idx)
        
        return heading

    def update_pointer(self, idx):
        '''
        Update the mem pointer to the back of the window
        mem_pointer = blimit
        :param idx:
        :return:
        '''
        self.mem_pointer += idx
        # in this case the upperpart is equal to the upper margin
        self.upper = self.window
        self.flimit = self.mem_pointer + self.upper
        self.blimit = self.mem_pointer

        if self.flimit > self.route_end:
            self.mem_pointer = self.route_end - self.window
            self.flimit = self.route_end
            self.blimit = self.route_end - self.window

    def update_mid_pointer(self, idx):
        '''
        Update the mem pointer to the middle of the window
        :param idx:
        :return:
        '''
        # Update memory pointer
        change = idx - self.lower
        self.mem_pointer += change

        # update upper an lower margins
        self.upper = int(round(self.window/2))
        self.lower = self.window - self.upper

        # Update the bounds of the window
        self.flimit = self.mem_pointer + self.upper
        self.blimit = self.mem_pointer - self.lower
        if self.flimit > self.route_end:
            self.mem_pointer = (self.route_end - self.window) + self.lower
            self.flimit = self.route_end
            self.blimit = self.route_end - self.window
        if self.blimit <= 0:
            # the mem pointer should be in the midle of the window
            self.mem_pointer = self.lower
            self.blimit = 0
            self.flimit = self.mem_pointer + self.window

    def check_w_size(self):
        self.window = self.route_end if self.window > self.route_end else self.window

    def get_agreement(self, window_headings):
        a = np.full(len(window_headings), 1)
        return cos_sim(a, window_headings)

    def consensus_heading(self, wind_headings, h):
        '''
        Calculates the agreement of the window headings.
        If the agreement is above the threshold the best heading is used
        otherwise the last heading is used.
        :param wind_headings:
        :param h:
        :return:
        '''
        if self.get_agreement(wind_headings) >= self.agreement_thresh:
            self.recovered_heading.append(h)
        elif len(self.recovered_heading) > 0:
            self.recovered_heading.append(self.recovered_heading[-1])
        else:
            self.recovered_heading.append(h)

    def average_heading2(self, h):
        '''
        Calculates the average of the last window heading and the current window heading
        :param h:
        :return:
        '''
        if len(self.recovered_heading) > 0:
            self.recovered_heading.append(mean_angle([h, self.recovered_heading[-1]]))
        else:
            self.recovered_heading.append(h)

    def average_headings(self, wind_heading):
        '''
        Calculates the average of all window headings
        :param wind_heading:
        :return:
        '''
        self.recovered_heading.append(mean_angle(wind_heading))

    def dynamic_window_fixed(self, best):
        '''
        Change the window size depending on the best img match gradient.
        If the last best img sim > the current best img sim the window grows
        and vice versa
        :param best:
        :return:
        '''
        # Dynamic window adaptation based on match gradient.
        if best > self.prev_match or self.window <= self.min_window:
            self.window += self.window_margin
        else:
            self.window -= self.window_margin
        self.prev_match = best
    
    def dynamic_window_rate(self, best):
        '''
        Change the window size depending on the current best and previous img match gradient. 
        Update the size by the dynamic_rate (percetage of the window size)
        :param best:
        :return:
        '''
        # Dynamic window adaptation based on match gradient.
        if best > self.prev_match or self.window <= self.min_window:
            self.window += round(self.window * self.dynamic_range)
        else:
            self.window -= round(self.window * self.dynamic_range)
        self.prev_match = best

    def dynamic_window_log_rate(self, best):
        '''
        Change the window size depending on the current best and previous img match gradient. 
        Update the size by log of the current window size
        :param best:
        :return:
        '''
        # Dynamic window adaptation based on match gradient.
        if best > self.prev_match or self.window <= self.min_window:
            self.window += round(self.min_window/np.log(self.window))
        else:
            self.window -= round(np.log(self.window))
        self.prev_match = best

    def dynamic_window_h2(self, h):
        '''
        Change the window size depending on the best heading gradient.
        If the difference between the last heading and the current heading is > self.deg_diff
        then the window grows and vice versa
        :param h:
        :return:
        '''
        diff = abs(h - self.recovered_heading[-1])
        if diff > self.deg_diff or self.window <= self.min_window:
            self.window += self.window_margin
        else:
            self.window -= self.window_margin

    def dynamic_window_h(self, wind_headings):
        '''
        The window grows if the window headings disagree and vice versa
        :param wind_headings:
        :return:
        '''
        if self.get_agreement(wind_headings) <= self.agreement_thresh or self.window <= self.min_window:
            self.window += self.window_margin
        else:
            self.window -= self.window_margin

    def get_rec_headings(self):
        return self.recovered_heading

    def get_index_log(self):
        return self.matched_index_log

    def get_window_log(self):
        return self.window_log

    def get_rsims_log(self):
        return self.logs

    def get_confidence(self):
        return self.confidence

    def get_window_sims(self):
        return self.window_sims

    def get_best_sims(self):
        return self.best_sims

    def get_window_headings(self):
        return self.window_headings

    def get_CMA(self):
        return self.CMA

    def get_name(self):
        return 'seq2seqA-SMW'
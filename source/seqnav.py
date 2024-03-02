from source.utils import pick_im_matcher, dot_dist, mae, rmse, cor_dist, rmf, seq2seqrmf, pair_rmf, cos_sim, mean_angle
from source.analysis import d2i_rmfs_eval
import numpy as np
import copy
from tqdm import tqdm
from collections import deque
from source.imgproc import Pipeline
from source.utils import *

class SequentialPerfectMemory:

    def __init__(self, route_images, matching, deg_range=(-180, 180), degree_shift=1, 
                window=20, dynamic_range=0.1, w_thresh=None, mid_update=True, 
                **kwargs):
        self.route_end = len(route_images)
        self.route_images = route_images
        # f:
        self.ims = np.array(route_images)
        self.ref_window = True
        self.mp = 0
        # f:#
        self.deg_range = deg_range
        self.deg_step = degree_shift
        self.degrees = np.arange(*deg_range)
        self.matcher = pick_im_matcher(matching)
        # if the dot product distance is used we need to make sure the images are standardized
        if self.matcher == dot_dist:
            pipe = Pipeline(normstd=True)
            self.route_images = pipe.apply(route_images)

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
        if window < 0:
            self.starting_window = abs(window)
            self.window = abs(window)
            self.adaptive = True
            self.upper = int(round(self.window/2))
            self.lower = self.window - self.upper
            self.mem_pointer = self.window - self.upper
            self.w_thresh =  w_thresh
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
    
    #TODO Need a better name for this function
    def reset_window(self, pointer):
        '''
        Resets the pointer assuming the window size is the same
        '''
        self.mem_pointer = pointer
        self.flimit = self.mem_pointer + self.upper
        self.blimit = self.mem_pointer - self.lower

        if self.flimit > self.route_end:
            self.mem_pointer = (self.route_end - self.window) + self.lower
            self.flimit = self.route_end
            self.blimit = self.route_end - self.window
        if self.blimit <= 0:
            self.mem_pointer = self.lower
            self.blimit = 0
            self.flimit = self.mem_pointer + self.window
    
    def set_mem_pointer(self, i: int):
        '''
        Resets the pointer assuming the window size may have changed
        Recalculates the upper and lower margins
        '''
        self.mem_pointer = i
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

    # quick positioning
    # def get_quick_location(self,query_img):
    #     # if memory pointer for ref window
    #     # mw_ims = self.ims*1
    #     # if self.mp >= 0:
    #         # bw_lim,fw_lim = np.clip(np.array([self.mp-10,self.mp+11]),0,len(self.ims))
    #         # mw_ims = self.ims[bw_lim:fw_lim,:,:]
    #         # print('memory pointer: {}, memory window: {}:{}'.format(self.mp,bw_lim,fw_lim))
    #         # print('est location (memory pointer): {}'.format(self.mp))
    #         # return self.mp
    #     # get aprox img location
    #     # basic (works only for route imgs)
    #     ii = self.ims - query_img
    #     ii_dif = np.abs(np.sum(ii,axis=(1,2)))
    #     ii_idx = np.where(ii_dif==np.min(ii_dif))[0]
    #     # also basic (a bit better, only for query imgs)
    #     ii_eq = np.where(ii==0,1,0)
    #     ii_zeros = np.sum(ii_eq,axis=(1,2))
    #     ii_idz = np.where(ii_zeros==np.max(ii_zeros))[0]
    #     if ii_idz.shape[0] == 1:
    #         ii_idz = ii_idz[0]
    #     else:
    #         # TODO 
    #         # print('more than one possible id loc')
    #         # print(ii_idz)
    #         # import pdb; pdb.set_trace()
    #         ii_idz = int(np.sum(ii_idz)/len(ii_idz))
    #     # print('idx: {}, idz: {}'.format(ii_idx,ii_idz))
    #     # plt.plot(ii_dif)
    #     # plt.plot(ii_zeros)
    #     # plt.show()
    #     # import pdb; pdb.set_trace()
    #     import pdb; pdb.set_trace()
    #     return ii_idz

    # # TODO: integrated, slower, but better?
    # def get_intx_location(self,query_img):
    #     # merged windows
    #     wbl,wfl = 0,11
    #     mw_cases = np.zeros(self.ims.shape[0]-wfl)
    #     for wi in range(self.ims.shape[0]-wfl):
    #         iim = self.ims[wbl+wi:wfl+wi] - query_img
    #         iim_merge = np.sum(iim,axis=0)
    #         iim_nz = np.where(iim_merge==0,0,1)
    #         qim_nz = query_img * iim_nz
    #         iim_ox = iim_merge + qim_nz
    #         iim_eq = np.where(iim==0,1,0)
    #         iim_idz = np.sum(iim_eq)
    #         mw_cases[wi] = iim_idz
    #         import pdb; pdb.set_trace()
    #     mw_idz = np.where(mw_cases==np.max(mw_cases))[0]
    #     print('mw idz: {}, window: {}:{}'.format(mw_idz,mw_idz,mw_idz+wfl))
    #     import pdb; pdb.set_trace()
    #     return mw_idz

    # get heading
    # def get_pos_heading(self,query_img):
    #     # quick location 
    #     img_loc = self.get_quick_location(query_img)
    #     # reference window
    #     ims_rw = self.ims*1
    #     if self.ref_window:
    #         bw_lim,fw_lim = np.clip(np.array([img_loc-10,img_loc+11]),0,len(self.ims))
    #         ims_rw = self.ims[bw_lim:fw_lim,:,:]
    #         # print('est loc: {}, ref window: {}:{}'.format(img_loc,bw_lim,fw_lim))
    #     # get heading
    #     rot_cases = np.zeros(360)
    #     rot_ids = []
    #     for ri in range(360):
    #         iir = ims_rw - rotate(ri,query_img) 
    #         iir_eq = np.where(iir==0,1,0) # eq: zeros
    #         iir_zeros = np.sum(iir_eq,axis=(1,2)) # count zeros
    #         iir_idz = np.where(iir_zeros==np.max(iir_zeros))[0]
    #         rot_cases[ri] = np.max(iir_zeros) # best case (from rw)
    #         rot_ids.append(iir_idz) # best id (for angle=ri)
    #     # best heading
    #     heading = np.where(rot_cases==np.max(rot_cases))[0]
    #     if heading.shape[0] == 1:
    #         heading = heading[0]
    #     else:
    #         # TODO (usually very close, eg.: 42,43)
    #         # print('more than one possible heading')
    #         # print(heading)
    #         # import pdb; pdb.set_trace()
    #         heading = np.min(heading)
    #     # best img id, according to best heading
    #     heading_id = rot_ids[heading]
    #     if heading_id.shape[0] == 1:
    #         heading_id = np.arange(bw_lim,fw_lim)[heading_id[0]]
    #     else:
    #         # TODO
    #         # print('more than id for heading')
    #         # print(heading_id)
    #         # import pdb; pdb.set_trace()
    #         heading_id = np.min(heading_id)
    #     # update memory pointer
    #     if self.mp >= 0:
    #         self.mp = heading_id
    #     # print('heading: {}, heading id: {}'.format(heading,heading_id))
    #     return img_loc,heading,heading_id

    # def get_heading_alt(self,query_img):
    #     # location in route
    #     qim_loc = self.get_img_location(query_img)
    #     imx = self.ims[qim_loc]
    #     # self.mp = qim_loc*1
    #     # get heading ("rmf")
    #     rot_cases = np.arange(360)
    #     for ri in range(360):
    #         imr = rotate(ri,imx)
    #         rx_dif = (imx-imr).nonzero()[0].shape[0]
    #         rot_cases[ri] = rx_dif
    #     # rmin_id = np.where(rot_cases==np.min(rot_cases))
    #     rmin_id = np.where(np.abs(rot_cases)==np.min(np.abs(rot_cases)))
    #     # print('rmin_id: {}'.format(rmin_id))
    #     rmin_deg = rot_cases[rmin_id]
    #     heading = np.min(rmin_deg) # same for all = min
    #     # print('heading: {}'.format(heading))
    #     return heading
            
    def segment_memory(self,threshold=2880):
        mem_ids = [0]
        mem_id = 0
        mode = 0
        # dif: neg: something appears, pos: dissapears
        while mem_id < self.ims.shape[0]:
            ii_dif = self.ims - self.ims[mem_id]
            ii_dif_nz = np.where(ii_dif>0,1,ii_dif)
            ii_dif_nz = np.where(ii_dif_nz<0,-1,ii_dif_nz)
            ii_dif_sum = np.sum(ii_dif_nz,axis=(1,2))

            if mode == 0:
                ii_idx = np.where(ii_dif_sum[mem_id:]<0)[0]
                idx = mem_id + ii_idx[0] if ii_idx.shape[0] > 0 else self.ims.shape[0]
                mem_ids.append(idx)
                mem_id = idx + 1
                # print(idx)
                # print(mem_ids)
                # print(ii_dif_sum.astype(int)[idx-2:idx+11])
                mode = 1
                # import pdb; pdb.set_trace()

            elif mode == 1:
                # print('\nmode 1')
                ii_dif_dif = ii_dif_sum - np.roll(ii_dif_sum,-1)
                ii_idz = np.where(ii_dif_dif[mem_id:]<-threshold)[0]
                idz = mem_id+1 + ii_idz[0] if ii_idz.shape[0] > 0 else self.ims.shape[0]
                ii_idx = np.where(ii_dif_dif[idz:]>0)[0]
                idx = idz + ii_idx[0] if ii_idx.shape[0] > 0 else self.ims.shape[0]
                # import pdb; pdb.set_trace()
                mem_ids.append(idx)
                mem_id = idx + 1
                # print(idx)
                # print(mem_ids)
                # print(ii_dif_dif.astype(int)[idx-2:idx+11])
                mode = 0
                # import pdb; pdb.set_trace()
        return mem_ids
        

    def get_pos_heading(self, qseq):
        # quick location
        self.mp = self.mem_pointer*1
        # merging
        qseq = np.array(qseq)
        ii0 = qseq[0] - qseq[2]
        ii1 = qseq[1] - qseq[2]
        z0 = np.where(ii0>0,1,ii0)
        z0 = np.where(z0<0,-1,z0)
        z1 = np.where(ii1>0,1,ii1)
        z1 = np.where(z1<0,-1,z1)
        zx = z0 * z1
        qx = qseq[2] * zx
        xims = self.ims * zx
        # heading from integration
        import pdb; pdb.set_trace()
        return [0,0,0]

    def get_heading(self, query_img):
        '''
        Recover the heading given a query image
        :param query_img:
        :return:
        '''
        # get the rotational similarities between a query image and a window of route images
        wrsims = rmf(query_img, self.route_images[self.blimit:self.flimit], self.matcher, self.deg_range, self.deg_step)

        # plot wrsims -> plot3d(wrsims)

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
        matched_idx = self.mem_pointer + (idx - self.lower)
        self.matched_index_log.append(matched_idx)

        #evaluate ridf
        # h_eval = self.eval_ridf(wrsims[idx])

        if self.adaptive:
            best = wind_sims[idx]
            # TODO here I need to make the updating function modular
            self.dynamic_window_log_rate(best)
            self.check_w_size()

        # Update memory pointer
        # if h_eval:
        self.update_mid_pointer(idx)
        # else:
        #     self.set_mem_pointer(self.mem_pointer + 1)

        # the heading changes if the rmf quality is low

        #heading = self.evaluated_heading(h_eval)
        # import pdb; pdb.set_trace()
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
        self.flimit = self.mem_pointer + self.upper
        self.blimit = self.mem_pointer

        if self.flimit > self.route_end:
            self.mem_pointer = self.blimit + idx
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

    def dynamic_window_sim(self, best):
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
        Change the window size d,subtitles=[]epending on the current best and previous img match gradient. 
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


    def navigate(self, query_imgs):
        assert isinstance(query_imgs, list)
        for query_img in query_imgs:
            self.get_heading(query_img)
        return self.recovered_heading, self.window_log
    
    # def navigate(self, query_imgs):
    #     assert isinstance(query_imgs, list)

    #     # upper = int(self.window/2)
    #     # lower = self.window - upper
    #     # mem_pointer = upper
    #     mem_pointer = 0
    #     flimit = self.window
    #     blimit = 0
    #     self.window_log.append([blimit, flimit])
    #     # For every query image
    #     for query_img in query_imgs:

    #         # get the rotational similarities between a query image and a window of route images
    #         wrsims = rmf(query_img, self.route_images[blimit:flimit], self.matcher, self.deg_range, self.deg_step)
    #         self.window_log.append([blimit, flimit])
    #         # Holds the best rot. match between the query image and route images
    #         wind_sims = []
    #         # Recovered headings for the current image
    #         wind_headings = []
    #         # get best similarity match adn index w.r.t degrees
    #         indices = self.argminmax(wrsims, axis=1)
    #         for i, idx in enumerate(indices):
    #             wind_sims.appe,subtitles=[]nd(wrsims[i, idx])
    #             wind_headings.append(self.degrees[idx])

    #         # Save the best degree and sim for each window similarities
    #         self.window_sims.append(wind_sims)
    #         self.window_headings.append(wind_headings)
    #         # append the rsims of all window route images for that current image
    #         self.logs.append(wrsims)
    #         idx = self.argminmax(wind_sims)
    #         self.best_sims.append(wind_sims[idx])
    #         h = wind_headings[idx]
    #         self.recovered_heading.append(h)
    #         # self.average_heading2(h)
    #         # self.average_headings(wind_headings)
    #         # self.consensus_heading(wind_headings, h)

    #         mem_pointer += idx
    #         if mem_pointer + self.window > self.route_end:
    #             mem_pointer = blimit + idx
    #             flimit = self.route_end
    #             blimit = self.route_end - self.window
    #         else:
    #             blimit = mem_pointer
    #             flimit = mem_pointer + self.window

    #         self.matched_index_log.append(mem_pointer)
    #         # self.window_log.append([blimit, flimit])


    #         # Change the pointer and bounds for an adaptive window.
    #         if self.adaptive:
    #             self.dynamic_window_sim(wind_sims[idx])
    #             # self.dynamic_window_h2(h)
    #             # self.dynamic_window_h(wind_headings)

    #         #
    #         # # Lower confidence of the memories depending on the match score
    #         # window_mean = sum(wind_sims)/len(wind_sims)
    #         # if i == 0: # If this is the first window
    #         #     self.CMA.extend([window_mean] * 2)
    #         # else:
    #         #     cma = self.CMA[-1]
    #         #     self.CMA.append(cma + ((window_mean-cma)/(len(self.CMA)+1)))
    #         # for j in range(mem_pointer, limit):
    #         #     if wind_sims[j-mem_pointer] > self.CMA[-1]:
    #         #         self.confidence[j] -= 0.1

    #     return self.recovered_heading, self.window_log

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
            pipe = Pipeline(normstd=True)
            self.route_images = pipe.apply(route_images)

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
        # self.matcher = pick_im_matcher(matching)
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

        if self.flimit > self.route_end:
            self.mem_pointer = (self.route_end - self.window) + self.lower
            self.flimit = self.route_end
            self.blimit = self.route_end - self.window
        if self.blimit <= 0:
            self.mem_pointer = self.lower
            self.blimit = 0
            self.flimit = self.mem_pointer + self.window

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
        heading = wind_headings[idx]
        self.recovered_heading.append(heading)

        # log the memory pointer before the update
        # mem_pointer - upper can cause the calc_dists() to go out of bounds
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
            self.mem_pointer = self.blimit + idx
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

    def dynamic_window_sim(self, best):
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
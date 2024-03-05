from source.utils import pick_im_matcher, mae, rmse, dot_dist, cor_dist, rmf, seq2seqrmf, pair_rmf, cos_sim, mean_angle
from source.analysis import d2i_rmfs_eval
import numpy as np
import copy
from collections import deque
from .navs import Navigator
from source.imgproc import Pipeline


class SequentialPerfectMemory(Navigator):

    def __init__(self, route_images, matcher='mae', deg_range=(-180, 180), degree_shift=1, 
                window=20, dynamic_range=0.1, w_thresh=None, mid_update=True, sma_size=3,
                **kwargs):
        super().__init__(route_images, matcher=matcher, deg_range=deg_range, degree_shift=degree_shift, **kwargs)
        
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

    def get_heading(self, query_img):
        '''
        Recover the heading given a query image
        :param query_img:
        :return:
        '''
        query_img = self.pipe.apply(query_img)
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


class Seq2SeqPerfectMemory(Navigator):
    
    def __init__(self, route_images, matcher='mae', deg_range=(-180, 180), degree_shift=1, 
                 w_thresh=None, mid_update=True, sma_size=3,
                 window=20, dynamic_range=0.1, queue_size=3, **kwargs):
        super().__init__(route_images, matcher=matcher, deg_range=deg_range, degree_shift=degree_shift, **kwargs)
        # current sequence params
        self.queue_size = queue_size
        self.queue = deque(maxlen=queue_size)

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
        # Matching variables
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

    def get_heading(self, query_img):
        '''
        Recover the heading given a query image
        :param query_img:
        :return:
        '''
        query_img = self.pipe.apply(query_img)
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
        matched_idx = self.blimit + idx
        self.matched_index_log.append(matched_idx)

        if self.adaptive:
            best = wind_sims[idx]
            # TODO here I need to make the updating function modular
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
        if self.adaptive:
            return f's2sA-SMW({self.starting_window})'
        else:
            return f's2sSMW({self.window})'
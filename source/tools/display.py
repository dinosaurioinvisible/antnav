import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from source.utils import check_for_dir_and_create
from source.utils import mae, rmf, rotate
from source.imgproc import Pipeline
from source.routedatabase import Route
sns.set_context("paper", font_scale=1)

def heading_res_and_rmfs(trial: dict, route: Route, pipe: Pipeline, rmf: callable, matcher: callable, trial_i_of_interest: list, deg_range=(-180, 180),
                         save_path=None):
    
    degrees = np.arange(*deg_range)
    route_id = route.get_route_id()
    # TODO: for stattic tets only.
    # for live test I need to add the trial images ionto the trail dict (or object?) and get the image from that
    ref_imgs = route.get_imgs()
    ref_imgs = pipe.apply(ref_imgs)
    trial_imgs = trial['q_imgs']
    trial_imgs = pipe.apply(trial_imgs)

    for ti in trial_i_of_interest:

        best_i = trial['min_dist_index'][ti]
        best_ref_im = ref_imgs[best_i]
        q_im = trial_imgs[ti]
        trial_matched_i = trial['matched_index'][ti]
        matched_ref_im = ref_imgs[trial_matched_i]
        print(f'trial image index {ti},', f'best optimal ref image {best_i},', f'trial matched index{trial_matched_i}' )

        #get the RIDFS, their minima and headings
        q_best_ridf = rmf(q_im, best_ref_im, matcher=matcher, d_range=deg_range)
        q_best_ridf_i = np.argmin(q_best_ridf)
        q_best_ridf_h = degrees[q_best_ridf_i]
        #######################################################
        q_matched_ridf = rmf(q_im, matched_ref_im, matcher=matcher, d_range=deg_range)
        q_matched_ridf_i = np.argmin(q_matched_ridf)
        q_matched_ridf_h = degrees[q_matched_ridf_i]
        
        # Roatate the image to the minima
        q_im_best_rot = rotate(q_best_ridf_h, q_im)
        q_im_matched_rot = rotate(q_matched_ridf_h, q_im)

        fig = plt.figure(figsize=(7, 6))
        #plt.suptitle('')
        rows = 4
        cols = 2

        # query img
        ax = fig.add_subplot(rows, 1, 1)
        ax.set_title(f'query image ({ti})')
        ax.imshow(q_im, cmap='gray')
        ax.set_axis_off()

        # opt match img
        ax = fig.add_subplot(rows, cols, 3)
        ax.set_title(f'optimal route image (route index ({best_i})')
        ax.imshow(best_ref_im, cmap='gray')
        ax.set_axis_off()

        # opt residual image
        ax = fig.add_subplot(rows, cols, 4)
        ax.set_title(f'optimal residual image')
        res_img = np.abs(q_im_best_rot - best_ref_im)
        ax.imshow(res_img, cmap='hot')
        ax.set_axis_off()

        # matched img
        ax = fig.add_subplot(rows, cols, 5)
        ax.set_title(f'matched route image (route index ({trial_matched_i}))')
        ax.imshow(matched_ref_im, cmap='gray')
        ax.set_axis_off()

        # matched residual image    
        ax = fig.add_subplot(rows, cols, 6)
        ax.set_title(f'matched residual image')
        res_img = np.abs(q_im_matched_rot - matched_ref_im)
        ax.imshow(res_img, cmap='hot')
        ax.set_axis_off()

        ax = fig.add_subplot(rows, 1, 4)
        ax.plot(degrees, q_best_ridf, label='optimal match RIDF')
        ax.plot(degrees, q_matched_ridf, label='matched RIDF')
        ax.set_xlabel('Degrees')
        ax.set_ylabel('MAE')
        plt.tight_layout()
        plt.legend()
        fig.savefig(os.path.join(save_path, f'aliasing-exp-trail_i({ti})-route({route_id}).png'))
        fig.savefig(os.path.join(save_path, f'aliasing-exp-trail_i({ti})-route({route_id}).pdf'))
        #plt.show()
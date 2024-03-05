import numpy as np
from scipy.spatial.distance import cdist
import os
import cv2 as cv
import matplotlib.pyplot as plt
from source.utils import rotate, pair_rmf, mae, mse, rmf, check_for_dir_and_create, weighted_mse
from source.display import plot_route_errors


def iqr_outliers(data):
    '''
    Calculates the inter-quartile outliers.
    Q3-Q1
    :param data:
    :return:
    '''
    q3 = np.percentile(data, 75)
    q1 = np.percentile(data, 25)
    iqr = q3 - q1
    outliers_over = data[data > (q3 + 1.5 * iqr)]
    outliers_under = data[data < (q1 - 1.5 * iqr)]
    out = np.append(outliers_over, outliers_under)
    return out


def perc_outliers(data):
    '''
    Calculates of the percetage of outlires as
    defined by the iqroutliers function
    :param data:
    :return:
    '''
    out = iqr_outliers(data)
    perc = len(out)/len(data)
    return perc


def log_error_points(route, traj, thresh=0.5, target_path=None, aw_agent=None):
    if not target_path:
        logs_path = 'route'
    logs_path = target_path
    check_for_dir_and_create(logs_path)
    # the antworld agent or query images for static bench
    if not route.get('qimgs') and aw_agent:
        agent = aw_agent()
    elif route.get('qimgs'):
        aw_agent = None 
    else:
        raise Exception('No query images and no agent')
    # get xy coords
    traj_xy = np.column_stack((traj['x'], traj['y']))
    route_xy = np.column_stack((route['x'], route['y']))

    rsims_matrices = traj['rmfs']
    # for the window version the structure of the RMF logs is 
    # dim0-> test points
    # dim1-> window rmfs
    # dim2-> the actual RMF (usualy 360 degrees of search angle)
    index_log = traj['matched_index']
    degrees = np.arange(*eval(traj['deg_range']))

    # Loop through every test point
    for i in range(len(traj['heading'])):
        #find the optimal/closest image match
        dist = np.squeeze(cdist(np.expand_dims(traj_xy[i], axis=0), route_xy, 'euclidean'))
        min_dist_i = np.argmin(dist)
        min_dist = dist[min_dist_i]
        # the index from the route that the agent matched best (the best match for this query image)
        route_match_i = index_log[i]
        point_ang_error = traj['errors'][i]
        # Analysis only for points that have a distance more than the threshold awayfrom the route
        if traj['errors'][i] >= thresh:
            point_path = os.path.join(logs_path, f'{i}-error={round(point_ang_error, 2)}')
            check_for_dir_and_create(point_path)
            # Save window images or 
            if traj.get('window_log'):
                w = traj.get('window_log')[i]
                for wi in range(w[0], w[1]):
                    imgfname = route['filename'][wi]
                    cv.imwrite(os.path.join(point_path, imgfname), route['imgs'][wi])
            else: # If perfect memory is used
                imgfname = route['filename'][route_match_i]
                cv.imwrite(os.path.join(point_path, imgfname) , route['imgs'][route_match_i])

            # save the minimum distance image
            imgfname = 'mindist-img{}.png'.format(min_dist_i)
            cv.imwrite(os.path.join(point_path, imgfname) , route['imgs'][min_dist_i])

            # save the best matched route image
            imgfname = 'matched-img{}.png'.format(route_match_i)
            cv.imwrite(os.path.join(point_path, imgfname) , route['imgs'][route_match_i])
            
            # Save the query image rotated to the extractred direction
            h = traj['heading'][i]
            if route.get('qimgs'):
                rimg = rotate(h, route['qimgs'][i])
                imgfname = 'rotated-grid-h' + str(h) + '.png'
                cv.imwrite(os.path.join(point_path, imgfname), rimg)
                imgfname = 'test-grid.png'
                cv.imwrite(os.path.join(point_path, imgfname), route['qimgs'][i])
            else:
                img = agent.get_img(traj_xy[i], h)
                imgfname = 'queryimg-matched-heading' + str(h) + '.png'
                cv.imwrite(os.path.join(point_path, imgfname), img)
            # Save ridf
            if traj.get('window_log'):
                w = traj.get('window_log')[i]
                window_index_of_route_match = route_match_i - w[0]
                rsim = rsims_matrices[i][window_index_of_route_match]
            else:
                rsim = rsims_matrices[i][route_match_i]
            fig = plt.figure()
            plt.plot(degrees, rsim)
            fig.savefig(os.path.join(point_path, 'rsim.png'))
            plt.close(fig)

            # Save window or full memory rsims heatmap
            # fig = plt.figure()
            # plt.imshow(rsims_matrices[i].tolist(), cmap='hot')
            # fig.savefig(os.path.join(point_path, 'heat.png'))
            # plt.close(fig)
            
            path = os.path.join(point_path, 'map.pdf')
            plot_route_errors(route, traj, route_i=route_match_i, error_i=i, path=path, size=(3, 3))

            

def rgb02nan(imgs, color=None):
    if not color:
        color = (0.0, 0.0, 0.0)
    nans = [np.nan, np.nan, np.nan]
    for i, img in enumerate(imgs):
        img = img.astype(np.float64)
        indices = np.where(np.all(img == color, axis=-1))

        for r, c in zip(indices[0], indices[1]):
            img[r, c, :] = nans
        imgs[i] = img
    return imgs


def nanrgb2grey(imgs):
    """
    Turn RGB images with NaNs to greyscale
    :param imgs:
    :return:
    """
    if isinstance(imgs, list):
        return [np.nanmean(img, axis=-1) for img in imgs]

    return np.nanmean(imgs, axis=-1)


def nanrbg2greyweighted(imgs):
    """
    Turn RGB images with NaNs to greyscale with weights for each channel
    :param imgs:
    :return:
    """
    rgb_weights = [0.2989, 0.5870, 0.1140]

    if isinstance(imgs, list):
        return [np.average(img, weights=rgb_weights, axis=-1) for img in imgs]

    return np.average(imgs, weights=rgb_weights, axis=-1)


def flip_gauss_fit(rsim, d_range=(-180, 180), eta=None):
    '''
    eta in the stutzl paper was set to 0.65
    '''
    degrees = np.arange(d_range[0], d_range[1])
    # get the mean of the function
    mu = degrees[np.argmin(rsim)]
    minimum = np.min(rsim)
    # depth of the RMF shape
    depth = np.max(rsim) - minimum
    if not eta:
        muidx = np.argmin(rsim)
        half_depth=depth/2
        maxindx=np.argmax(rsim[0:muidx])
        p1 = np.argmin(np.abs((rsim[maxindx:muidx]-half_depth)),0)
        p1=maxindx+p1
        maxindx=np.argmax(rsim[muidx:])
        p2 = np.argmin(np.abs(rsim[muidx:muidx+maxindx]-half_depth),0)
        p2 = muidx + p2
        eta = np.abs(p2-p1)

    # delta angles from the mean 
    d_angles = degrees-mu
    # fit the flipped gaussian to the RMF
    g_fit = depth*(1 - np.exp(-(d_angles**2)/(2*(eta**2)))) + minimum
    return g_fit

def gauss_curve(rsim, d_range=(-180, 180), eta=None):
    degrees = np.arange(d_range[0], d_range[1])
    # get the mean of the function
    mu = degrees[np.argmin(rsim)]
    minimum = np.min(rsim)
    # depth of the RMF shape
    depth = np.max(rsim) - minimum
    if not eta:
        muidx = np.argmin(rsim)
        half_depth=depth/2
        maxindx=np.argmax(rsim[0:muidx])
        p1 = np.argmin(np.abs((rsim[maxindx:muidx]-half_depth)),0)
        p1=maxindx+p1
        maxindx=np.argmax(rsim[muidx:])
        p2 = np.argmin(np.abs(rsim[muidx:muidx+maxindx]-half_depth),0)
        p2 = muidx + p2
        eta = np.abs(p2-p1)
    # delta angles from the mean 
    d_angles = degrees-mu
    # standard gaussian mirroring the RMF
    # add 1 to the gaussian in order to avoid having zero values for the weights
    g_fit = depth*(np.exp(-(d_angles**2)/(2*(eta**2)))) + minimum + 1
    return g_fit

def eval_pair_rmf(imgs, d_range=(-180, 180)):
    rsims = pair_rmf(imgs, imgs, d_range=d_range)
    fit_errors = np.empty(len(imgs))
    for i, rsim in enumerate(rsims):
        g_curve = flip_gauss_fit(rsim, d_range=d_range)
        err = mse(rsim, g_curve)
        fit_errors[i] = err
    return fit_errors

def eval_rmf_fit(ref_img, imgs, d_range=(-180, 180)):
    rsims = rmf(ref_img, imgs, d_range=d_range)
    fit_errors = []
    for rsim in rsims:
        g_curve = flip_gauss_fit(rsim, d_range=d_range)
        err = mse(rsim, g_curve)
        fit_errors.append(err)
    return fit_errors

def eval_gauss_rmf_fit(rsims, d_range=(-180, 180), weighted=False):
    fit_errors = []
    for rsim in rsims:
        g_curve = flip_gauss_fit(rsim, d_range=d_range)
        if weighted:
            w = gauss_curve(rsim, d_range)
            err = weighted_mse(rsim, g_curve, w)
        else:
            err = mse(rsim, g_curve)
        fit_errors.append(err)
    return fit_errors

def d2i_eval(imgs, d_range=(-180, 180)):
    rsims = pair_rmf(imgs, imgs, d_range=d_range)
    depths = np.max(rsims, axis=1)-np.min(rsims, axis=1)
    integs = np.trapz(rsims, axis=1)
    return depths/integs

def d2i_rmfs_eval(rsims):
    rsims = np.array(rsims)
    if len(rsims.shape) < 2:
        rsims = np.expand_dims(rsims, axis=0)
    depths = np.max(rsims, axis=1)-np.min(rsims, axis=1)
    integs = np.trapz(rsims, axis=1)
    return np.squeeze(depths/integs)


def trans_catch_areas(query_img, ref_imgs, matcher=mae):
        '''
        Find the translational catchment areas for each RIDF between the query image and the ref images.
        '''
        ridf_field = rmf(query_img, ref_imgs, matcher=matcher, d_range=(-180, 180))

        # translational idf of the ridf field minima
        tidf = np.min(ridf_field, axis=1)
        # the minima in translation
        min_tidf_i = np.argmin(tidf)
        diffs = np.diff(tidf)

        halfright = diffs[min_tidf_i:]
        halfleft = diffs[:min_tidf_i]
        #find the poit where the gradiend sign change moving away from the minima
        # add 1 cause the diff array is one elment shorter than the tidf
        right_lim = min_tidf_i + np.argmax(halfright < 0.0) + 1
        # flip the half left side of the RIDF in order to find the first
        # possitive change fo the gradient moving or the minima to the left
        left_lim = min_tidf_i - np.argmax(np.flip(halfleft) > 0.0)
        area_lims = (left_lim, right_lim)
        area = right_lim - left_lim

        # plt.plot(tidf)
        # left_lim = area_lims[0]
        # right_lim = area_lims[1]
        # plt.scatter(range(left_lim, right_lim), tidf[left_lim:right_lim])
        # plt.show()

        return ridf_field, area, area_lims


def catch_areas(query_img, ref_imgs, matcher=mae):
    '''
    Find the catchment areas for each RIDF between the query image and the ref images.
    '''
    ridf_field = rmf(query_img, ref_imgs, matcher=matcher, d_range=(-180, 180))
    indices = np.argmin(ridf_field, axis=1)
    #grad = np.gradient(ridf_field, axis=1)
    diffs = np.diff(ridf_field, axis=1)
    areas = np.empty(len(ref_imgs))
    area_lims = []
    for i, j in enumerate(indices):
        halfright = diffs[i, j:]
        halfleft = diffs[i, :j]
        #find the poit where the gradiend sign change moving away from the minima
        right_lim = j + np.argmax(halfright < 0.0)
        # flip the half left side of the RIDF in order to find the first
        # possitive change fo the gradient moving or the minima to the left
        left_lim = j - np.argmax(np.flip(halfleft) > 0.0)
        area_lims.append((left_lim, right_lim))
        areas[i] = right_lim - left_lim
    return ridf_field, areas, area_lims
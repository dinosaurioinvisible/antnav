import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2 as cv
import math
import os
import shutil
from scipy.spatial.distance import cosine, correlation, cdist, pdist
from scipy.stats import circmean
from collections.abc import Iterable


def display_image(image, size=(10, 10), title='Title', save_id=None):
    """
    Display the image given as a 2d or 3d array of values.
    :param size: Size of the plot for the image
    :param image: Input image to display
    """
    image = np.squeeze(image)
    fig = plt.figure(figsize=size)
    plt.imshow(image, cmap='gray', interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # or plt.axis('off')
    plt.title(title, loc="left")
    plt.tight_layout(pad=0)
    if save_id: fig.savefig(str(save_id) + ".png", bbox_inches="tight")
    plt.show()


def display_split(image_l, image_r, size=(10, 10), file_name=None):
    image_l = np.squeeze(image_l)
    image_r = np.squeeze(image_r)

    fig = plt.figure(figsize=size)
    fig.add_subplot(1, 2, 1)
    plt.imshow(image_l, cmap='gray', interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    fig.add_subplot(1, 2, 2)
    plt.imshow(image_r, cmap='gray', interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # or plt.axis('off')
    if file_name: plt.savefig(file_name + ".png")
    plt.show()


def save_image(path, img):
    cv.imwrite(path, img)# , cmap='gray')


def plot_route(route, traj=None, scale=None, window=None, windex=None, save=False, size=(10, 10), path=None, title=None,
               ax=None, label=None):
    '''
    Plots the route and any given test points if available.
    Note the route headings are rotated 90 degrees as the 0 degree origin
    for the antworld is north but for pyplot it is east.
    :param route:
    :param traj:
    :param scale:
    :param window:
    :param windex:
    :param save:
    :param size:
    :param path:
    :param title:
    :return:
    '''
    if not ax:
        fig, ax = plt.subplots(figsize=size)
    ax.set_title(title,  loc="left")
    
    u, v = pol2cart_headings(90 - route['yaw'])
    ax.scatter(route['x'], route['y'], label='training route')
    ax.annotate('Start', (route['x'][0], route['y'][0]))
    #ax.quiver(route['x'], route['y'], u, v, scale=scale)
    if window is not None and windex:
        start = window[0]
        end = window[1]
        ax.quiver(route['x'][start:end], route['y'][start:end], u[start:end], v[start:end], color='r', scale=scale)
        if not traj:
            ax.scatter(route['qx'][:windex], route['qy'][:windex])
        else:
            ax.scatter(traj['x'][:windex], traj['y'][:windex])
            u, v = pol2cart_headings(90 - traj['heading'])
            ax.quiver(traj['x'][:windex], traj['y'][:windex], u[:windex], v[:windex], scale=scale)
    # Plot grid test points
    if 'qx' in route and window is None and not traj:
        ax.scatter(route['qx'], route['qy'])
    # Plot the trajectory of the agent when repeating the route
    if traj and not window:
        # TODO: This re-correction (90 - headings) of the heading may not be necessary.
        # TODO: I need to test if this will work as expected when the new results are in.
        u, v = pol2cart_headings(90 - traj['heading'])
        ax.scatter(traj['x'], traj['y'], label=label)
        # ax.plot(traj['x'], traj['y'])
        ax.quiver(traj['x'], traj['y'], u, v, scale=scale)
    ax.set_aspect('equal', 'datalim')
    
    
    if save and windex:
        fig.tight_layout()
        fig.savefig(path + '/' + str(windex) + '.png')
        plt.close(fig)
    elif save:
        fig.tight_layout()
        #path = os.path.join(path, 'routemap.png')
        print(f'fig saved at: {path}')
        fig.savefig(path)
        plt.close(fig)
    return ax
    


def animated_window(route, window, traj=None, path=None, scale=70, save=False, size=(10, 10), title=None):
    check_for_dir_and_create(path)
    if path:
        save = True
    for i, w in enumerate(window):
        plot_route(route, traj=traj, window=w, windex=i, save=save, scale=scale, size=size, path=path, title=title)


def plot_map(world, route_cords=None, grid_cords=None, size=(10, 10), save=False, zoom=(), zoom_factor=1500,
             route_headings=None, grid_headings=None, error_indexes=None, marker_size=5, scale=40, route_zoom=False, save_id=None, window=None,
             path='', show=True, title=None):
    '''
    Plots a top down view of the grid world, with markers or quivers of route and grid positions
    :param world: Top down image of the world
    :param route_cords: X and Y route coordinates
    :param grid_cords: X and Y grid coordinates
    :param size: size of the figure
    :param save: If to save the image
    :param zoom: x and y tuple of zoom centre
    :param zoom_factor: Magnitude of zoom. (lower values is greater zoom)
    :param route_headings: Heading in degrees of the route positions
    :param grid_headings: Heading in degrees of grid positions
    :param marker_size: Size of route of grid marker
    :param scale: Size of quiver scale. (relative to image size)
    :param route_zoom: Rectangle zoom around the route
    :param save_id: A file id to save the plot and avoid override of the saved file
    :param window: The pointer to the memory window
    :return: -
    '''
    fig = plt.figure(figsize=size)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.xlabel('x coordinates', fontsize=14, fontweight='bold')
    plt.ylabel('y coordinates', fontsize=13, fontweight='bold')
    # Plot circles for route image locations
    if route_cords and route_headings is None: plt.plot(route_cords[0], route_cords[1],
                                                        marker="o", markersize=marker_size, linewidth=1, color='blue')
    # Plot stars for grid image locations
    if grid_cords and grid_headings is None: plt.scatter(grid_cords[0][0:save_id], grid_cords[1][0:save_id],
                                                         marker="*", s=marker_size, color='red')
    # Plot route images heading vectors
    if route_headings is not None:
        route_U, route_V = pol2cart_headings(90.0 - np.array(route_headings))
        plt.quiver(route_cords[0], route_cords[1], route_U, route_V, scale=scale, color='b')
    # Plot world grid images heading vectors
    # The variable save_id is used here to plot the vectors that have been matched with a window so far
    if grid_headings is not None and error_indexes is None:
        grid_U, grid_V = pol2cart_headings(90.0 - np.array(grid_headings))
        plt.quiver(grid_cords[0][0:save_id], grid_cords[1][0:save_id],
                                grid_U[0:save_id], grid_V[0:save_id], scale=scale, color='r')
    if error_indexes:
        grid_U, grid_V = pol2cart_headings(90.0 - np.array(grid_headings))
        plt.quiver(grid_cords[0], grid_cords[1], grid_U, grid_V, scale=scale, color='b')
        error_headings = [grid_headings[i] for i in error_indexes]
        error_X = [grid_cords[0][i] for i in error_indexes]
        error_Y = [grid_cords[1][i] for i in error_indexes]
        error_U, error_V = pol2cart_headings(90.0 - np.array(error_headings))
        plt.quiver(error_X, error_Y, error_U, error_V, scale=scale, color='r')

    # Plot window vectors only
    if window:
        window = range(window[0], window[1])
        route_U, route_V = pol2cart_headings(90.0 - np.array(route_headings))
        plt.quiver([route_cords[0][i] for i in window], [route_cords[1][i] for i in window],
                   [route_U[i] for i in window], [route_V[i] for i in window], scale=scale, color='c')

    plt.imshow(world, zorder=0, extent=[-0.158586 * 1000, 10.2428 * 1000, -0.227704 * 1000, 10.0896 * 1000])
    if zoom:
        plt.xlim([zoom[0] - zoom_factor, zoom[0] + zoom_factor])
        plt.ylim([zoom[1] - zoom_factor, zoom[1] + zoom_factor])
    if route_zoom:
        # plt.ylim([])
        plt.xlim([4700, 6500])
    plt.xticks([]), plt.yticks([])
    # plt.title("A", loc="left", fontsize=20)
    plt.tight_layout(pad=0)
    if save and save_id: fig.savefig(path + 'graph' + str(save_id) + '.png')
    if save and not save_id: fig.savefig(path)
    if show: plt.show()
    if save: plt.close()


def load_route_naw(path, route_id=1, imgs=False, query=False, max_dist=0.5, grid_path=None):
    if not grid_path and query:
        # TODO: Need to edit the function so that is receive the parent path of the route directory
        grid_path = os.path.dirname(path)
        grid_path = os.path.dirname(grid_path)
        grid_path = os.path.dirname(grid_path) + '/grid70'
    route_data = pd.read_csv(path + 'route' + str(route_id) + '.csv', index_col=False)
    route_data = route_data.to_dict('list')
    # convert the lists to numpy arrays
    for k in route_data:
        route_data[k] = np.array(route_data[k])
    if imgs:
        imgs = []
        for i in route_data['filename']:
            img = cv.imread(path + i, cv.IMREAD_GRAYSCALE)
            imgs.append(img)
        route_data['imgs'] = imgs

    # Sample positions and images from the grid near the route for testing
    if query:
        grid = pd.read_csv(grid_path + '/grid.csv')
        grid = grid.to_dict('list')
        for k in grid:
            grid[k] = np.array(grid[k])

        grid_xy = np.transpose(np.array([grid['x'], grid['y']]))
        query_indexes = np.empty(0, dtype=int)
        qx = []
        qy = []
        qimg = []
        # Fetch images from the grid that are located nearby route images.
        # for each route position
        for i, (x, y) in enumerate(zip(route_data['x'], route_data['y'])):
            # get distance between route point and all grid points
            dist = np.squeeze(cdist([(x, y)], grid_xy, 'euclidean'))
            # indexes of distances within the limit
            indexes = np.where(dist <= max_dist)[0]
            # check which indexes have not been encountered before
            mask = np.isin(indexes, query_indexes, invert=True)
            # get the un-encountered indexes
            indexes = indexes[mask]
            # save the indexes
            query_indexes = np.append(query_indexes, indexes)

            for i in indexes:
                qx.append(grid_xy[i, 0])
                qy.append(grid_xy[i, 1])
                imgfile = os.path.join(grid_path, grid['filename'][i])
                qimg.append(cv.imread(imgfile, cv.IMREAD_GRAYSCALE))

        route_data['qx'] = np.array(qx)
        route_data['qy'] = np.array(qy)
        route_data['qimgs'] = qimg

    return route_data


def write_route(path, route, route_id=1):
    route = pd.DataFrame(route)
    route.to_csv(os.path.join(path, 'route' + str(route_id) + '.csv'), index=False)


def travel_dist(x, y):
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    steps = np.sqrt(dx**2+dy**2)
    return np.sum(steps)


def calc_dists(route, indexa, indexb):
    assert len(indexa) == len(indexb)
    xa = route['x'][indexa]
    ya = route['y'][indexa]

    xb = route['x'][indexb]
    yb = route['y'][indexb]

    dx = xa - xb
    dy = ya - yb
    return np.sqrt(dx**2+dy**2)


def load_route(route_id, grid_pos_limit=200):
    # Path/ Directory settings
    antworld_path = '../AntWorld'
    route_id = str(route_id)
    route_id_dir = '/ant1_route' + route_id + '/'
    route_dir = antworld_path + route_id_dir
    grid_dir = antworld_path + '/world5000_grid/'

    # World top down image
    world = mpimg.imread(grid_dir + 'world5000_grid.png')

    # Grid Images
    data = pd.read_csv(grid_dir + 'world5000_grid.csv', header=0)
    data = data.values

    # Route
    route_data = pd.read_csv(route_dir + 'ant1_route' + route_id + '.csv', header=0)
    route_data = route_data.values

    ## Organize data
    # Grid data
    X = data[:, 0]  # x location of the image in the world_grid
    Y = data[:, 1]  # y location of the image in the world_grid
    img_path = data[:, 4]  # Name of the image file

    # Route data
    X_route = route_data[:, 0].tolist()  # x location of the image in the route
    Y_route = route_data[:, 1].tolist()  # y location of the image in the route
    Heading_route = route_data[:, 3]     # Image heading
    imgs_route_path = route_data[:, 4]  # Name of the image file

    # Load route images
    max_norm = 1
    route_images = []
    for i in range(0, len(imgs_route_path)):
        img = cv.imread(route_dir + imgs_route_path[i][1:], cv.IMREAD_GRAYSCALE)
        # Normalize
        #img = img * max_norm / img.max()
        route_images.append(img)

    # Load world grid images
    max_norm = 1
    X_inlimit = []
    Y_inlimit = []
    world_grid_imgs = []

    # Fetch images from the grid that are located nearby route images.
    for i in range(0, len(X_route)):
        dist = []
        for j in range(0, len(X)):
            d = (math.sqrt((X_route[i] - X[j]) ** 2 + (Y_route[i] - Y[j]) ** 2))
            dist.append(d)
            if grid_pos_limit > d:  # Maximum distance limit from the Route images
                X_inlimit.append(X[j])
                Y_inlimit.append(Y[j])
                X[j] = 0
                Y[j] = 0
                img_dir = grid_dir + img_path[j][1:]
                img = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
                # Normalize
                # img = img * max_norm / img.max()
                world_grid_imgs.append(img)

    return world, X_inlimit, Y_inlimit, world_grid_imgs, X_route, Y_route, Heading_route, route_images


def line_incl(x, y):
    '''
    Calculates the inclination of lines defined by 2 subsequent coordinates
    :param x:
    :param y:
    :return:
    '''
    incl = np.arctan2(np.subtract(y[1:], y[:-1]), np.subtract(x[1:], x[:-1])) * 180 / np.pi
    incl = np.append(incl, incl[-1])
    return incl


def meancurv2d(x, y):
    '''
    Calculates the mean curvature of a set of points (x, y) that belong to a curve.
    :param x:
    :param y:
    :return:
    '''
    # first derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)

    # second derivatives
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # calculate the mean curvature from first and second derivatives
    curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5

    return np.mean(curvature)


def squash_deg(degrees):
    '''
    Squashes degrees into the range of 0-360
    This is useful when dealing with negative degrees or degrees over 360
    :param degrees: A numpy array of values in degrees
    :return:
    '''
    assert not isinstance(degrees, list)
    return degrees % 360


def mean_angle(angles):
    angles = np.deg2rad(angles)
    return np.rad2deg(circmean(angles))


def pol2cart(r, theta):
    '''
    Coverts polar coordinates to cartesian coordinates
    :param r: An array or single value of radial values
    :param theta: An array or single values ot angles theta
    :return:
    '''
    x = np.multiply(r, np.cos(theta))
    y = np.multiply(r, np.sin(theta))
    return x, y


def pol2cart_headings(headings):
    """
    Convert degree headings to U,V cartesian coordinates
    :param headings: list of degrees
    :return: 2D coordinates
    """
    rads = np.radians(headings)
    U, V = pol2cart(1, rads)
    return U, V


def pre_process(imgs, sets):
    """
    Gaussian blur, edge detection and image resize
    :param imgs:
    :param sets:
    :return:
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    if sets.get('shape'):
        shape = sets['shape']
        imgs = [cv.resize(img, shape, interpolation=cv.INTER_NEAREST) for img in imgs]
    if sets.get('blur'):
        imgs = [cv.GaussianBlur(img, (5, 5), 0) for img in imgs]
    if sets.get('edge_range'):
        lims = sets['edge_range']
        imgs = [cv.Canny(img, lims[0], lims[1]) for img in imgs]

    return imgs if len(imgs) > 1 else imgs[0]


def image_split(image, overlap=None, blind=0):
    '''
    Splits an image to 2 parts, left and right part evenly when no overlap is provided.
    :param image: Image to split. 2 dimentional ndarray
    :param overlap: Degrees of overlap between the 2 images
    :param blind: Degrees of blind visual field at the back of the agent
    :return:
    '''
    num_of_cols = image.shape[1]
    if blind:
        num_of_cols_perdegree = int(num_of_cols / 360)
        blind_pixels = blind * num_of_cols_perdegree
        blind_pixels_per_side = int(blind_pixels/2)
        image = image[:, blind_pixels_per_side:-blind_pixels_per_side]

    num_of_cols = image.shape[1]
    split_point = int(num_of_cols / 2)
    if overlap:
        num_of_cols_perdegree = int(num_of_cols / (360-blind))
        pixel_overlap = overlap * num_of_cols_perdegree
        l_split = split_point + int(pixel_overlap/2)
        r_split = split_point - int(pixel_overlap/2)
        left = image[:, :l_split]
        right = image[:, r_split:]
    else:
        left = image[:, :split_point]
        right = image[:, split_point:]

    return left, right


def rotate(d, image):
    """
    Converts the degrees into columns and rotates the image.
    Positive degrees rotate the image clockwise
    and negative degrees rotate the image counter clockwise
    :param d: number of degrees the agent will rotate its view
    :param image: An np.array that we want to shift.
    :return: Returns the rotated image.
    """
    assert abs(d) <= 360

    num_of_cols = image.shape[1]
    num_of_cols_perdegree = num_of_cols / 360
    cols_to_shift = int(round(d * num_of_cols_perdegree))
    return np.roll(image, -cols_to_shift, axis=1)


def center_ridf(ridfs):
    '''
    Cneter the ridfs so that the minima are 
    in the middle of the array.
    '''
    for i, ridf in enumerate(ridfs):
        idx = np.argmin(ridf)
        center_shift = int(round(-idx + len(ridf)/2))
        ridfs[i] = np.roll(ridf, center_shift)
    return ridfs


def mse(a, b):
    """
    Image Differencing Function MSE
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [np.mean(np.subtract(ref_img, a)**2) for ref_img in b]

    return np.mean(np.subtract(a, b)**2)


def weighted_mse(a, b, weights=None):
    if isinstance(b, list):
        return [np.mean(weights * np.subtract(ref_img, a)**2) for ref_img in b]

    return np.mean(weights * np.subtract(a, b)**2)


def rmse(a, b):
    """
    Image Differencing Function RMSE
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [np.sqrt(np.mean(np.subtract(ref_img, a)**2)) for ref_img in b]

    return np.sqrt(np.mean(np.subtract(a, b)**2))


def mae(a, b):
    """
    Image Differencing Function MAE
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [np.mean(np.abs(a - img)) for img in b]

    return np.mean(np.abs(a - b))


def nanmae(a, b):
    """
    Image Differencing Function MAE for images with nan values
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [np.nanmean(np.abs(a - img)) for img in b]

    return np.nanmean(np.abs(a - b))


def cov(a, b):
    """
    Calculates covariance (non sample)
    Assumes flattened arrays
    :param a:
    :param b:
    :return:
    """
    assert len(a) == len(b)

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    return np.sum((a - a_mean) * (b - b_mean)) / (len(a))


def cor_coef(a, b):
    """
    Calculate the correlation coefficient
    :param a: A single image or vector
    :param b: A single image or vector
    :return:
    """
    a = a.flatten()
    b = b.flatten()
    return cov(a, b) / (np.std(a) * np.std(b))


def cor_dist(a, b):
    """
    Calculates the correlation coefficient distance
    between a (list of) vector(s) b and reference vector a
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [_cc_dist(a, img) for img in b]

    return _cc_dist(a, b)

def _cc_dist(a, b):
    """
    Calculates the correlation coefficient distance
    between two vectors.
    :param a:
    :param b:
    :return:
    """
    amu = np.mean(a)
    bmu = np.mean(b)
    a = a - amu
    b = b - bmu
    ab = np.mean(a * b)
    avar = np.mean(np.square(a))
    bvar = np.mean(np.square(b))
    return 1.0 - ab / np.sqrt(avar * bvar)


def nan_correlation_dist(a, b):
    """
    Calculates the correlation coefficient distance
    between two vectors.
    Where a and b can contain nan values.
    :param a:
    :param b:
    :return:
    """
    amu = np.nanmean(a)
    bmu = np.nanmean(b)
    a = a - amu
    b = b - bmu
    ab = np.nanmean(a * b)
    avar = np.nanmean(np.square(a))
    bvar = np.nanmean(np.square(b))
    dist = 1.0 - ab / np.sqrt(avar * bvar)
    return dist


def nan_cor_dist(a, b):
    """
    Calculates the correlation coefficient distance
    between a (list of) vector(s) b and reference vector a.
    Where a and b can contain nan values.
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [nan_correlation_dist(a, img) for img in b]

    return nan_correlation_dist(a, b)


# def cos_dist(a, b):
#     """
#     Calculates cosine similarity
#     between a (list of) vector(s) b and reference vector a
#     :param a:
#     :param b:
#     :return:
#     """
#     if isinstance(b, list):
#         return [cosine(a, img) for img in b]

#     return cosine(a, b)

def cos_dist(a, b):
    """
    Cossine Distance
    :param a:
    :param b:
    :return:
    """
    if isinstance(b, list):
        return [1.0 - (np.vdot(a, img) / (norm(a) * norm(img))) for img in b]
    
    return 1.0 - (np.vdot(a, b) / (norm(a) * norm(b)))


def cos_sim(a, b):
    """
    Cossine similarity.
    :param a:
    :param b:
    :return:
    """
    return np.vdot(a, b) / (norm(a) * norm(b))


def dot_dist(a, b):
    """
    Returns the dot product distance.
    This function assumes the vectors have zero means and unit variance.
    :param a: numpy vector or matrix 
    :param b: numpy vector or matrix 
    :return: distance between [0, 2]
    """
    if isinstance(b, list):
        return [1 - np.vdot(a, img) for img in b]
    
    return 1 - np.vdot(a, b)


def entropy_im(img, bins=256):
    #get the histogram
    amarg = np.histogramdd(np.ravel(img), bins = bins)[0]/img.size
    amarg = amarg[np.ravel(amarg) > 0]
    return -np.sum(np.multiply(amarg, np.log2(amarg)))


def mutual_inf(a, b, bins=256):
    if isinstance(b, list):
        return [_mut_inf(a, img, bins) for img in b]

    return mutual_inf(a, b, bins)


def _mut_inf(a, b, bins=256):
    hist_2d, x_edges, y_edges = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    pab = hist_2d / float(np.sum(hist_2d))
    pa = np.sum(pab, axis=1) # marginal for x over y
    pb = np.sum(pab, axis=0) # marginal for y over x
    pa_pb = pa[:, None] * pb[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pab > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pab[nzs] * np.log(pab[nzs] / pa_pb[nzs]))


def entropy_dist(a, b, bins=256):
    if isinstance(b, list):
        return [_entropy_dist(a, img, bins) for img in b]

    return _entropy_dist(a, b, bins)


def _entropy_dist(a, b, bins=256):
    ## how many bins? 256 always? 
    # ask andy here for join entropy vs H(a)
    # amarg = np.histogram(np.ravel(a), bins = bins)[0]/a.size
    # amarg = amarg[amarg > 0]
    # aentropy = -np.sum(np.multiply(amarg, np.log2(amarg)))

    hist_2d, x_edges, y_edges = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    pab = hist_2d / float(np.sum(hist_2d))
    pa = np.sum(pab, axis=1) # marginal for a over b
    pb = np.sum(pab, axis=0) # marginal for b over a
    pa_pb = pa[:, None] * pb[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pab > 0 # Only non-zero pab values contribute to the sum

    pab_joint = pab[np.logical_and(pab, pab)]
    ent_pab = -np.sum(pab_joint * np.log(pab_joint))
    #here from practical rasons i could also subtract from the entropy of a
    return ent_pab - (np.sum(pab[nzs] * np.log(pab[nzs] / pa_pb[nzs])))


def pick_im_matcher(im_matcher=None):
    matchers = {'corr': cor_dist, 'dot': dot_dist, 'rmse': rmse, 'mse':mse, 'mae': mae, 'entropy':entropy_dist}
    if not matchers.get(im_matcher):
        raise Exception('Non valid matcher method name')
    return matchers.get(im_matcher)

def rmf(query_img, ref_imgs, matcher=mae, d_range=(0, 360), d_step=1):
    """
    Rotational Matching Function.
    Rotates a query image and compares it with one or more reference images
    :param query_img:
    :param ref_imgs:
    :param matcher:
    :param d_range:
    :param d_step:
    :return:
    """
    assert d_step > 0
    assert not isinstance(query_img, list)
    if not isinstance(ref_imgs, list):
        ref_imgs = [ref_imgs]

    degrees = range(*d_range, d_step)
    total_search_angle = round((d_range[1] - d_range[0]) / d_step)
    sims = np.empty((len(ref_imgs), total_search_angle), dtype=np.float)

    for i, rot in enumerate(degrees):
        # rotated query image
        rqimg = rotate(rot, query_img)
        sims[:, i] = matcher(rqimg, ref_imgs)

    return sims if sims.shape[0] > 1 else sims[0]


def pair_rmf(query_imgs, ref_imgs, matcher=mae, d_range=(0, 360), d_step=1):
    """
    Pairwise Rotational Matching Function
    :param query_imgs:
    :param ref_imgs:
    :param matcher:
    :param d_range:
    :param d_step:
    :return:
    """
    assert d_step > 0
    assert isinstance(query_imgs, list)
    assert isinstance(ref_imgs, list)
    assert len(query_imgs) == len(ref_imgs)

    degrees = range(*d_range, d_step)
    total_search_angle = round((d_range[1] - d_range[0]) / d_step)
    sims = np.empty((len(ref_imgs), total_search_angle), dtype=np.float)

    for i, (q_img, r_img) in enumerate(zip(query_imgs, ref_imgs)):
        sims[i] = [matcher(rotate(rot, q_img), r_img) for rot in degrees]

    return sims

def seq2seqrmf(query_imgs, ref_imgs, matcher=mae, d_range=(0, 360), d_step=1):
    """
    Rotational Matching Function.
    Rotates multiple query images and compares then with one or more reference images
    :param query_img:
    :param ref_imgs:
    :param matcher:
    :param d_range:
    :param d_step:
    :return:
    """
    assert d_step > 0
    assert isinstance(query_imgs, Iterable)
    if not isinstance(ref_imgs, list):
        ref_imgs = [ref_imgs]

    degrees = range(*d_range, d_step)
    total_search_angle = round((d_range[1] - d_range[0]) / d_step)
    sims = np.empty((len(query_imgs)*len(ref_imgs), total_search_angle), dtype=np.float)
    for i, query_img in enumerate(query_imgs):
        for j, rot in enumerate(degrees):
            # rotated query image
            rqimg = rotate(rot, query_img)
            sims[:(i+1)*len(query_imgs), j] = matcher(rqimg, ref_imgs)

    return sims


def flatten_imgs(imgs):
    assert isinstance(imgs, list)
    return [img.flatten() for img in imgs]


def cross_corr(sub_series, series):
    return [np.dot(s, sub_series) for s in series]


def degree_error_logs(x_cords, y_cords, x_route_cords, y_route_cords, route_heading, recovered_headings, degree_thresh=30):
    k = []  # Holds the position of the memory with the shortest distance to the wg position
    logs = {'x_route': [], 'y_route': [], 'route_heading': [], 'route_idx': [],
            'x_grid': [], 'y_grid': [], 'grid_heading': [], 'grid_idx': [], 'errors': []}
    route_end = len(x_route_cords)
    search_step = 10
    memory_pointer = 0
    limit = memory_pointer + search_step
    for i in range(0, len(x_cords)):  # For every grid position
        distance = []
        for j in range(memory_pointer, limit):  # For every route position
            d = math.sqrt((x_cords[i] - x_route_cords[j]) ** 2 + ((y_cords[i] - y_route_cords[j]) ** 2))
            distance.append(d)
        k.append(distance.index(min(distance)) + memory_pointer)
        memory_pointer = k[-1]
        limit = memory_pointer + search_step
        if limit > route_end: limit = route_end
        error = (180 - abs(abs(recovered_headings[i] - route_heading[k[-1]]) - 180))
        if error > degree_thresh or error < -degree_thresh:
            logs['x_route'].append(x_route_cords[k[-1]])
            logs['y_route'].append(y_route_cords[k[-1]])
            logs['route_heading'].append(route_heading[k[-1]])
            logs['route_idx'].append(k[-1])
            logs['x_grid'].append(x_cords[i])
            logs['y_grid'].append(y_cords[i])
            logs['grid_heading'].append(recovered_headings[i])
            logs['grid_idx'].append(i)
            logs['errors'].append(error)
    return logs


def degree_error(x_cords, y_cords, x_route_cords, y_route_cords, route_heading, recovered_headings):
    k = []  # Holds the index of the memory with the shortest distance to the grid position
    errors = []  # Holds the error between the world grid image and the closest route image
    route_end = len(x_route_cords)
    search_step = 10
    memory_pointer = 0
    limit = memory_pointer + search_step
    # TODO: Need to iterate the total number of window comparisons instead of the total coord points
    #   they are not equal for current window seq. algorithms
    for i in range(0, len(x_cords)):  # For every grid position
        distance = []
        for j in range(memory_pointer, limit):  # For every route position
            d = math.sqrt((x_cords[i] - x_route_cords[j]) ** 2 + ((y_cords[i] - y_route_cords[j]) ** 2))
            distance.append(d)
        k.append(distance.index(min(distance)) + memory_pointer)
        errors.append(180 - abs(abs(recovered_headings[i] - route_heading[k[-1]]) - 180))
        memory_pointer = k[-1]
        limit = memory_pointer + search_step
        if limit > route_end: limit = route_end
    return errors, k


def seq_angular_error(route, trajectory, memory_pointer=0, search_step=20):
    # TODO: Modify the function to calculate all the distances first (distance matrix)
    # TODO: and then calculate the minimum argument and extract the error.
    # Holds the angular error between the query position and the closest route position
    errors = []
    mindist_index = []
    route_end = len(route['x'])
    search_step = search_step
    memory_pointer = memory_pointer
    limit = memory_pointer + search_step

    grid_xy = np.column_stack([trajectory['x'], trajectory['y']])
    route_xy = np.column_stack([route['x'], route['y']])
    recovered_headings = trajectory['heading']
    route_heading = route['yaw']

    # For every query position
    for i in range(len(trajectory['heading'])):
        # get distance between route point and all grid points
        xy = route_xy[memory_pointer:limit]
        dist = np.squeeze(cdist(np.expand_dims(grid_xy[i], axis=0), xy, 'euclidean'))
        idx = np.argmin(dist)
        mindist_index.append(idx + memory_pointer)
        errors.append(180 - abs(abs(recovered_headings[i] - route_heading[mindist_index[-1]]) - 180))
        memory_pointer = mindist_index[-1]
        # update the limit
        limit = memory_pointer + search_step
        if limit > route_end: limit = route_end
    return errors, mindist_index


def angular_error(route, trajectory):
    '''
    route yaw and trajectory yaw must be within 
    '''
    # Holds the angular error between the query position and the closest route position
    errors = []
    mindist_index = []
    grid_xy = np.column_stack([trajectory['x'], trajectory['y']])
    route_xy = np.column_stack([route['x'], route['y']])
    recovered_headings = trajectory['heading']
    route_heading = route['yaw']

    # For every query position
    for i in range(len(trajectory['heading'])):
        # get distance between route point and all grid points
        dist = np.squeeze(cdist(np.expand_dims(grid_xy[i], axis=0), route_xy, 'euclidean'))
        idx = np.argmin(dist)
        mindist_index.append(idx)
        errors.append(180 - abs(abs(recovered_headings[i] - route_heading[mindist_index[-1]]) - 180))
    return errors, mindist_index

def angular_diff(a, b):
    '''
    Assumes angles are in degrees in [-inf, inf]
    return: smallest angle diff in [0, 180]
    '''
    assert len(a) == len(b)
    return np.abs(180 - np.abs((a - b)%360 - 180))


def divergence_traj(route, trajectory):
    traj_xy = np.column_stack([trajectory['x'], trajectory['y']])
    route_xy = np.column_stack([route['x'], route['y']])
    #calculate all pairs distances
    dists = cdist(traj_xy, route_xy, metric='euclidean')
    # return the min distance for each test position.
    return np.amin(dists, axis=1)


def mean_seq_angular_error(route, trajectory):
    errors, _ = seq_angular_error(route, trajectory)
    return np.mean(errors)


def mean_angular_error(route, trajectory):
    errors, _ = angular_error(route, trajectory)
    return np.mean(errors)


def mean_degree_error(x_cords, y_cords, x_route_cords, y_route_cords, route_heading, recovered_headings):
    error, k = degree_error(x_cords, y_cords, x_route_cords, y_route_cords, route_heading, recovered_headings)
    return sum(error) / len(error)


def check_for_dir_and_create(directory, remove=False):
    if remove and os.path.exists(directory):
        shutil.rmtree(directory)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def load_loop_route(route_dir, route_id=1, grid_pos_limit=100):
    # Path/ Directory settings
    route_id_dir = 'route_' + str(route_id) + '/'
    csv_file = 'route_' + str(route_id) + '.csv'
    route_dir = route_dir + route_id_dir
    grid_dir = '/home/efkag/PycharmProjects/ant_world_alg_bench/AntWorld/world5000_grid/'

    # World top down image
    world = mpimg.imread(grid_dir + 'world5000_grid.png')

    # Grid Images
    data = pd.read_csv(grid_dir + 'world5000_grid.csv', header=0)
    data = data.values

    # Route
    route_data = pd.read_csv(route_dir + csv_file, header=0)
    route_data = route_data.values

    ## Organize data
    # Grid data
    X = data[:, 0]  # x location of the image in the world_grid
    Y = data[:, 1]  # y location of the image in the world_grid
    img_path = data[:, 4]  # Name of the image file

    # Route data
    X_route = route_data[:, 0].tolist()  # x location of the image in the route
    Y_route = route_data[:, 1].tolist()  # y location of the image in the route
    Heading_route = route_data[:, 3]  # Image heading
    imgs_route_path = route_data[:, 4]  # Name of the image file

    # Load route images
    max_norm = 1
    route_images = []
    for i in range(0, len(imgs_route_path)):
        temp = route_dir + imgs_route_path[i]
        img = cv.imread(route_dir + imgs_route_path[i][1:], cv.IMREAD_GRAYSCALE)
        # Normalize
        #img = img * max_norm / img.max()
        route_images.append(img)

    # Load world grid images
    max_norm = 1
    X_inlimit = []
    Y_inlimit = []
    world_grid_imgs = []

    # Fetch images from the grid that are located nearby route images.
    for i in range(0, len(X_route), 1):
        for j in range(0, len(X), 1):
            d = (math.sqrt((X_route[i] - X[j]) ** 2 + (Y_route[i] - Y[j]) ** 2))
            if 1 < d < grid_pos_limit:  # Maximum distance limit from the Route images
                X_inlimit.append(X[j])
                Y_inlimit.append(Y[j])
                X[j] = 0
                Y[j] = 0
                img_dir = grid_dir + img_path[j][1:]
                img = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
                # Normalize
                # img = img * max_norm / img.max()
                world_grid_imgs.append(img)

    return world, X_inlimit, Y_inlimit, world_grid_imgs, X_route, Y_route, Heading_route, route_images


def scale2_0_1(a):
    return (a-np.min(a))/(np.max(a)-np.min(a))
import os
import numpy as np
import pandas as pd
import copy
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from source.utils import pol2cart_headings
import seaborn as sns


fig_records_keys = ['fig_path', 'script_path', 'date']
def fig_record(records_filepath, fig_path='', script_path='', date='', **kwargs):
    # if the records file already exists
    if os.path.isfile(records_filepath):
        df = pd.read_csv(records_filepath)
        records = df.to_dict('list')
        records['fig_path'].append(fig_path)
        records['script_path'].append(script_path)
        records['date'].append(date)
    else:
        #TODo: might need to handle that in the future with a creation of a file
        # make disct ...
        raise RuntimeError('Records file does not exist')



def nans_imgshow(img):
    if np.nanmax(img) > 1.:
        img = img/np.nanmax(img)
    f = plt.figure()
    ax = f.add_subplot(111)
    masked_array = np.ma.array(img, mask=np.isnan(img))
    cmap = copy.copy(matplotlib.cm.jet)
    cmap.set_bad('white', 1.)
    ax.imshow(masked_array, interpolation='nearest', cmap=cmap)
    plt.show()


def imgshow(image, size=(10, 10), title='', path='', save_id=None):
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
    if path and save_id:
        save_path = os.path.join(path, str(save_id) + ".png") 
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

def imghist(img, bins=256, ax=None):
    if ax:
        ax.hist(img.flatten(), bins=bins, density=True)
        return ax
    else:
        plt.hist(img.flatten(), bins=bins, density=True)
        plt.show()


def plot_3d(data, show=True, rows_cols_idx=111, title='', save=False, path=''):
    '''
    Plots the 2d data given in a 3d wireframe.
    Assumes first dimension is number of images,
    second dimension is search angle.
    :param data:
    :return:
    '''
    if not path:
        path='3dplot.png'
    # The second dimension of the data is the search angle
    # i.e the degree rotated to the left (-deg) and degree rotated to the right (+deg)
    deg = round(data.shape[1]/2)
    no_of_imgs = data.shape[0]

    x = np.linspace(-deg, deg, deg*2)
    y = np.linspace(0, no_of_imgs, no_of_imgs)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(rows_cols_idx, projection='3d')
    # ax = plt.subplot(rows_cols_idx, projection='3d')
    ax.plot_surface(X, Y, data, cmap=cm.coolwarm)
    ax.title.set_text(title)
    ax.set_ylabel('Images in transaltion (image index)')
    ax.set_xlabel('Search angle in degrees')
    ax.set_zlabel('Distance measure', rotation=45)
    ax.view_init(azim=45)
    if save: fig.savefig(path)    

    if show: plt.show()

def img_3d(img, show=True, title=''):
    x, y = img.shape

    x = np.arange(0, x)
    y = np.arange(0, y)
    X, Y = np.meshgrid(y, x)


    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(X, Y, img)
    if show: plt.show()


def plot_ridf_multiline(data, scatter=False, labels=None, xlabel=None, ylabel=None):
    if data.ndim < 2:
        data = np.expand_dims(data, axis=0)

    deg = round(data.shape[1] / 2)
    # no_of_imgs = data.shape[0]
    x = np.linspace(-deg, deg, deg * 2)
    for i, line in enumerate(data):
        plt.plot(x, line, label=labels[i])
        if scatter: plt.scatter(x, line)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend()
    plt.show()


def plot_multiline(data, scatter=False, labels=None, xlabel=None, ylabel=None):
    if data.ndim < 2:
        data = np.expand_dims(data, axis=0)

    for i, line in enumerate(data):
        plt.plot(line, label=labels[i])
        if scatter: plt.scatter(range(len(line)), line)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

    
def plot_route_errors(route, traj, route_i, error_i, size=(10, 10), scale=None, path=None):
    fig, ax = plt.subplots(figsize=size)
    u, v = pol2cart_headings(90 - route['yaw'])
    ax.scatter(route['x'], route['y'])
    ax.annotate('Start', (route['x'][0], route['y'][0]))
    #ax.quiver(route['x'], route['y'], u, v, scale=scale)

    u, v = pol2cart_headings(90 - route['yaw'][route_i])
    ax.quiver(route['x'][route_i], route['y'][route_i], u, v, scale=scale, label='match', color='y')

    u, v = pol2cart_headings(90 - traj['heading'][error_i])
    ax.quiver(traj['x'][error_i], traj['y'][error_i], u, v, scale=scale, label='test pos', color='r')
    # Plot the window memories
    if traj['window_log']:
        window = traj['window_log'][error_i]
        start = window[0]
        end = window[1]
        u, v = pol2cart_headings(90 - route['yaw'][start:end])
        ax.quiver(route['x'][start:end], route['y'][start:end], u, v, color='r', scale=70)

    plt.legend()
    plt.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    else:
        plt.show()

def plot_matches(route, traj, matches, scale=None, size=(10, 10), path=None, title=None):
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title,  loc="left")
    
    # Plot the route datapoints
    u, v = pol2cart_headings(90 - route['yaw'])
    ax.scatter(route['x'], route['y'])
    ax.quiver(route['x'], route['y'], u, v, scale=scale)
    #Plot the trajectory
    u, v = pol2cart_headings(90 - traj['heading'])
    ax.scatter(traj['x'], traj['y'])
    # ax.plot(traj['x'], traj['y'])
    ax.quiver(traj['x'], traj['y'], u, v, scale=scale)

    # plot match lines
    rx = route['x'][matches]
    ry = route['y'][matches]
    tx = traj['x']
    ty = traj['y']
    xs = np.column_stack((rx, tx))
    ys = np.column_stack((ry, ty))
    for x, y in zip(xs, ys):
        plt.plot(x, y, c='k')
    plt.tight_layout()
    if path:
        fig.savefig(path)
    plt.show()
    plt.close(fig)
    

def plot_route(route, traj=None, scale=None, window=None, windex=None, save=False, size=(10, 10), path=None, title=None):
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
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title,  loc="left")
    
    u, v = pol2cart_headings(90 - route['yaw'])
    ax.scatter(route['x'], route['y'])
    ax.quiver(route['x'], route['y'], u, v, scale=scale)
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
    if 'qx' in route and window is None:
        ax.scatter(route['qx'], route['qy'])
    # Plot the trajectory of the agent when repeating the route
    if traj and not window:
        # TODO: This re-correction (90 - headings) of the heading may not be necessary.
        # TODO: I need to test if this will work as expected when the new results are in.
        u, v = pol2cart_headings(90 - traj['heading'])
        ax.scatter(traj['x'], traj['y'])
        # ax.plot(traj['x'], traj['y'])
        ax.quiver(traj['x'], traj['y'], u, v, scale=scale)
    #plt.axis('equal')
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.tight_layout()
    if save and windex:
        fig.savefig(path + '/' + str(windex) + '.png')
        plt.close(fig)
    elif save:
        fig.savefig(path)

    if not save: plt.show()
    plt.close(fig)

def plot_ftl_route(route, traj=None, scale=None, window=None, windex=None, save=False, size=(10, 10), path=None, title=None):
    '''
    Plots the route and any given test points if available.
    Note the route headings are rotated 90 degrees as the 0 degree origin
    for the FTL seesm to be east but for some reason it does not plot properly
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
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title,  loc="left")
    
    u, v = pol2cart_headings(90 + route['yaw'])
    ax.scatter(route['x'], route['y'])
    ax.quiver(route['x'], route['y'], u, v, scale=scale)
    if window is not None and windex:
        start = window[0]
        end = window[1]
        ax.quiver(route['x'][start:end], route['y'][start:end], u[start:end], v[start:end], color='r', scale=scale)
        if not traj:
            ax.scatter(route['qx'][:windex], route['qy'][:windex])
        else:
            ax.scatter(traj['x'][:windex], traj['y'][:windex])
            u, v = pol2cart_headings(90 + traj['heading'])
            ax.quiver(traj['x'][:windex], traj['y'][:windex], u[:windex], v[:windex], scale=scale)
    # Plot grid test points
    if 'qx' in route and window is None:
        ax.scatter(route['qx'], route['qy'])
    # Plot the trajectory of the agent when repeating the route
    if traj and not window:
        u, v = pol2cart_headings(90 + traj['heading'])
        ax.scatter(traj['x'], traj['y'])
        # ax.plot(traj['x'], traj['y'])
        ax.quiver(traj['x'], traj['y'], u, v, scale=scale)
    plt.axis('equal')
    plt.tight_layout()
    if save and windex:
        fig.savefig(path + '/' + str(windex) + '.png')
        plt.close(fig)
    elif save:
        fig.savefig(path)

    if not save: plt.show()
    plt.close(fig)


def plot_ftl_route(route, traj=None, scale=None, window=None, windex=None, save=False, size=(10, 10), path=None, title=None):
    '''
    Plots the route and any given test points if available.
    Note the route headings are rotated 90 degrees as the 0 degree origin
    for the FTL seesm to be east but for some reason it does not plot properly
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
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title,  loc="left")
    
    u, v = pol2cart_headings(route['yaw'])
    ax.scatter(route['x'], route['y'], label='training')
    ax.quiver(route['x'], route['y'], u, v, scale=scale)
    if window is not None and windex:
        start = window[0]
        end = window[1]
        ax.quiver(route['x'][start:end], route['y'][start:end], u[start:end], v[start:end], color='r', scale=scale)
        if not traj:
            ax.scatter(route['qx'][:windex], route['qy'][:windex])
        else:
            ax.scatter(traj['x'][:windex], traj['y'][:windex])
            u, v = pol2cart_headings(traj['heading'])
            ax.quiver(traj['x'][:windex], traj['y'][:windex], u[:windex], v[:windex], scale=scale)
    # Plot grid test points
    if 'qx' in route and window is None:
        ax.scatter(route['qx'], route['qy'])
    # Plot the trajectory of the agent when repeating the route
    if traj and not window:
        u, v = pol2cart_headings(traj['heading'])
        ax.scatter(traj['x'], traj['y'], label='trial')
        # ax.plot(traj['x'], traj['y'])
        ax.quiver(traj['x'], traj['y'], u, v, scale=scale)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    if save and windex:
        fig.savefig(path + '/' + str(windex) + '.png')
        plt.close(fig)
    elif save:
        fig.savefig(path)

    if not save: plt.show()
    plt.close(fig)


def _plot_route_axis(route, ax, traj=None, scale=None, window=None, windex=None, size=(10, 10), title=None, **kwargs):
    ax.set_title(title,  loc="left")
    u, v = pol2cart_headings(90 - route['yaw'])
    ax.scatter(route['x'], route['y'])
    ax.quiver(route['x'], route['y'], u, v, scale=scale)
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
    if 'qx' in route and window is None:
        ax.scatter(route['qx'], route['qy'])
    # Plot the trajectory of the agent when repeating the route
    if traj and not window:
        # TODO: This re-correction (90 - headings) of the heading may not be necessary.
        # TODO: I need to test if this will work as expected when the new results are in.
        u, v = pol2cart_headings(90 - traj['heading'])
        ax.scatter(traj['x'], traj['y'])
        # ax.plot(traj['x'], traj['y'])
        ax.quiver(traj['x'], traj['y'], u, v, scale=scale)
    ax.set_aspect('equal')
    return ax


def plot_multiroute(routes: list, **kwargs):
    route_count = len(routes)
    fig, axes = plt.subplots(1, route_count)
    for ax, route in zip(axes, routes):
        route_dict = route.get_route_dict()    
        ax = _plot_route_axis(route_dict, ax, **kwargs)
    fig.tight_layout(pad=0)
    plt.show()


def heat_with_marginals(data, ax=None):
    '''
    Plot a 2D Heatmap with marginal distibutions.
    '''
    R, C = data.shape
    r = np.linspace(0, R, num=R)
    c = np.linspace(0, C, num=C)
    X, Y = np.meshgrid(r, c)
    x = X.flatten()
    y = Y.flatten()
    z = data.flatten()
    df = {'rows': x, 'cols': y, 'rimg': z}
    df = pd.DataFrame(df)
    g = sns.jointplot(data=df, x='rows', y='cols', kind="hist", ax=ax)
    
    #g = sns.jointplot(data=df, x='rows', y='cols', kind='hist', bins=(c, H))
    g.ax_marg_y.cla()
    g.ax_marg_x.cla()
    sns.heatmap(data=data, ax=g.ax_joint, cbar=False, cmap='Blues')

    row_marg = np.sum(data, axis=0)
    col_marg = np.sum(data, axis=1)
    g.ax_marg_y.barh(np.arange(0.5, R), col_marg, color='navy')
    g.ax_marg_x.bar(np.arange(0.5, C), row_marg, color='navy')

    # g.ax_joint.set_xticks(np.arange(0.5, R))
    # g.ax_joint.set_xticklabels(range(1, R + 1), rotation=0)
    # g.ax_joint.set_yticks(np.arange(0.5, C))
    # g.ax_joint.set_yticklabels(range(C), rotation=0)

    # remove ticks between heatmao and histograms
    g.ax_marg_x.tick_params(axis='x', bottom=False, labelbottom=False)
    g.ax_marg_y.tick_params(axis='y', left=False, labelleft=False)
    # remove ticks showing the heights of the histograms
    g.ax_marg_x.tick_params(axis='y', left=False, labelleft=False)
    g.ax_marg_y.tick_params(axis='x', bottom=False, labelbottom=False)

    g.fig.set_size_inches(17, 8)  # jointplot creates its own figure, the size can only be changed afterwards
    # g.fig.subplots_adjust(hspace=0.3) # optionally more space for the tick labels
    g.fig.subplots_adjust(hspace=0.05, wspace=0.02)  # less spaced needed when there are no tick labels
    return g

import math
from source.utils import calc_dists, line_incl, pol2cart_headings, check_for_dir_and_create, pol2cart, squash_deg, travel_dist
from matplotlib import pyplot as plt
import numpy as np
pi = math.pi


def binomial(i, n):
    """Binomial coefficient"""
    return math.factorial(n) / float(math.factorial(i) * math.factorial(n - i))


def bernstein(t, i, n):
    """Bernstein polynom"""
    return binomial(i, n) * (t ** i) * ((1 - t) ** (n - i))


def bezier(t, points):
    """Calculate coordinate of a point in the bezier curve"""
    n = len(points) - 1
    x = y = 0
    for i, pos in enumerate(points):
        bern = bernstein(t, i, n)
        x += pos[0] * bern
        y += pos[1] * bern
    return x, y


def bezier_curve_range(n, points):
    """Range of points in a curve bezier"""
    for i in range(n):
        t = i / float(n - 1)  # type:float
        yield bezier(t, points)


def bezier_curve_vert(n, verts):
    x_cord = []
    y_cord = []
    for i in range(n):
        t = i / float(n - 1)
        x, y = bezier(t, verts)
        x_cord.append(x)
        y_cord.append(y)
    return x_cord, y_cord


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


def random_circle_points(r, no_of_points=10):
    '''
    Generates random points within a circle given the desired radius.
    It assumes the center is (0,0)
    :param r:
    :param no_of_points:
    :return:
    '''
    r = r * np.sqrt(np.random.rand(no_of_points))
    theta = np.random.rand(no_of_points) * 2 * pi

    x, y = pol2cart(r, theta)
    xy = np.column_stack((x, y))
    return xy


def random_gauss_points(mean=(0, 0), sigma=10, no_of_points=5):
    '''

    :param mean:
    :param sigma:
    :param no_of_points:
    :return:
    '''
    mean = np.array(mean)
    cov = np.array([[sigma, 0], [0, sigma]])
    return np.random.multivariate_normal(mean, cov, no_of_points)


def random_line_points(start=-10, end=10, sigma=5, no_of_points=5):
    x = np.linspace(start, end, no_of_points)
    y = np.full_like(x, x)
    y[1:-1] = x[1:-1] + np.random.normal(0, sigma, no_of_points-2)
    xy = np.column_stack((x, y))
    return xy


def generate_from_points(path, generator='gauss', **kwargs):
    curve_points = kwargs['curve_points']
    if generator == 'points':
        points = np.genfromtxt(path + 'points.csv', delimiter=',')
    elif generator == 'gauss':
        mean = kwargs['mean']
        sigma = kwargs['sigma']
        points = random_gauss_points(mean, sigma)
    elif generator == 'circle':
        r = kwargs['r']
        points = random_circle_points(r)
    elif generator == 'line':
        start = kwargs['start']
        end = kwargs['end']
        sigma = kwargs['sigma']
        points = random_line_points(start, end, sigma)
    else:
        raise Exception('Provide a valid generator method')
    return generate(points, path, curve_points=curve_points)


def generate(xy, save_path, curve_points=250, plot=False):
    '''
    Given control points from one of the methods above.
    Uses the control points to generate points on a bezier curve.
    :param xy:
    :param save_path:
    :param curve_points:
    :param plot:
    :return:
    '''
    route = {}
    check_for_dir_and_create(save_path)
    np.savetxt(save_path + "points.csv", X=xy, delimiter=',')

    x_route, y_route = bezier_curve_vert(curve_points, xy)
    # 90 - rotates the origin of the degrees by 90 degrees counter clokwise
    # setting the origin (0 degrees) at the north
    heading = 90 - line_incl(x_route, y_route)
    heading = squash_deg(heading)

    route['x'] = np.array(x_route)
    route['y'] = np.array(y_route)
    #fixed z (elevation)
    z = 1.5
    route['z'] = np.full(curve_points, z)
    route['yaw'] = heading
    route['pitch'] = np.zeros(curve_points)
    route['roll'] = np.zeros(curve_points)

    print('mean curvature:', meancurv2d(x_route, y_route))
    print('traveled distance', travel_dist(route['x'], route['y']))
    print('average dist between points', travel_dist(route['x'], route['y'])/curve_points)
    if plot:
        u, v = pol2cart_headings(90 - heading)
        plt.scatter(x_route, y_route)
        plt.quiver(x_route, y_route, u, v, scale=60)
        plt.scatter(xy[:, 0], xy[:, 1])
        plt.show()
    return route


def generate_grid(steps):
    grid = {}
    xv = np.linspace(-10, 10, steps)
    yv = np.linspace(-10, 10, steps)

    x, y = np.meshgrid(xv, yv)

    x = x.flatten()
    y = y.flatten()

    grid['x'] = x
    grid['y'] = y
    z =1.5
    grid['z'] = np.full(steps*steps, z)
    grid['yaw'] = np.full(steps*steps, 0)
    grid['pitch'] = np.full(steps * steps, 0)
    grid['roll'] = np.full(steps * steps, 0)

    return grid


# generate_grid(100)

# x, y = random_circle_points(10, 6)
# plt.scatter(x, y)
# plt.show()
# route_id = 6
# generate([0, 0], '../new-antworld/route' + str(route_id) + '/', no_of_points=4, curve_points=200, route_id=route_id)

#
# np.random.seed(10)
# mean = np.array([0, 0])
# cov = np.array([[1, 10], [10, 1]])
# xy = np.random.multivariate_normal(mean, cov, 6)
#
# points = [
#     [0., 0.],
#     [0.5, 0.5],
#     [1.0, 1.0],
#     [2.5, 2.5],
# ]
# # Show control points
# xy = np.array(points)
#
# np.savetxt('../XYZbins/new_points_3.csv', xy, delimiter=',')
# plt.plot(xy[:, 0], xy[:, 1])
# plt.scatter(xy[:, 0], xy[:, 1])
#
# x_route, y_route = bezier_curve_vert(100, xy)
# heading = 90 - line_incl(x_route, y_route)
# # heading = line_incl(x_route, y_route)
# u, v = pol2cart_headings(90 - heading)
# # plt.plot(x_route, y_route)
# plt.scatter(x_route, y_route)
# plt.quiver(x_route, y_route, u, v, scale=50)
# plt.show()
#
# # Save to file
# z = 1.5
# path = '../XYZbins/new_route_3.csv'
# data = np.array([x_route, y_route, [z]*len(x_route), heading])
#
# # data.tofile(path)
# np.savetxt(path, data, delimiter=',')
#
# print(meancurv2d(x_route, y_route))

import os
import numpy as np
import cv2 as cv
import pandas as pd
from scipy.spatial.distance import cdist
from source.utils import calc_dists, squash_deg, travel_dist, pre_process, angular_error, seq_angular_error, travel_dist, meancurv2d
from source.unwraper import Unwraper
from source.imgproc import resize


class Route:
    def __init__(self, path, route_id, read_imgs=True, grid_path=None, max_dist=0.2):
        self.path = path
        self.grid_path = grid_path
        self.read_imgs = read_imgs
        self.max_dist = max_dist
        self.proc_imgs = []
        self.proc_qimgs= []
        self.route_id = str(route_id)
        self.route_dict = self.load_route()
        self.points_on_route = len(self.route_dict['x'])
        self.route_end = len(self.route_dict['x'])
        self.is_segmented = False
        self.segment_indices = None
        self.no_of_segments = None

    def load_route(self):
        data_path = os.path.join(self.path, 'route' + self.route_id + '.csv')
        route_data = pd.read_csv(data_path, index_col=False)
        route_data = route_data.to_dict('list')
        # convert the lists to numpy arrays
        for k in route_data:
            route_data[k] = np.array(route_data[k])
        if self.read_imgs:
            imgs = []
            for i in route_data['filename']:
                imgfile = os.path.join(self.path, i)
                img = cv.imread(imgfile, cv.IMREAD_GRAYSCALE)
                imgs.append(img)
            route_data['imgs'] = imgs

        # Sample positions and images from the grid near the route for testing
        if self.grid_path:
            grid = pd.read_csv(self.grid_path + '/grid.csv')
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
                indexes = np.where(dist <= self.max_dist)[0]
                # check which indexes have not been encountered before
                mask = np.isin(indexes, query_indexes, invert=True)
                # get the un-encountered indexes
                indexes = indexes[mask]
                # save the indexes
                query_indexes = np.append(query_indexes, indexes)

                for i in indexes:
                    qx.append(grid_xy[i, 0])
                    qy.append(grid_xy[i, 1])
                    imgfile = os.path.join(self.grid_path, grid['filename'][i])
                    qimg.append(cv.imread(imgfile, cv.IMREAD_GRAYSCALE))

            route_data['qx'] = np.array(qx)
            route_data['qy'] = np.array(qy)
            route_data['qyaw'] = np.full_like(route_data['qx'], 0.0)
            route_data['qimgs'] = qimg

        return route_data

    def imgs_pre_proc(self, params):
        if self.route_dict.get('qimgs'):
            self.proc_qimgs = pre_process(self.route_dict['qimgs'], params)
        self.proc_imgs = pre_process(self.route_dict['imgs'], params)
        return self.proc_imgs

    def segment_route(self, segment_size_m):
        self.is_segmented = True
        dist = travel_dist(self.route_dict['x'], self.route_dict['y'])
        self.no_of_segments = int(round(dist / segment_size_m))

        # xs = np.array_split(self.route_dict['x'], no_of_segments)
        # ys = np.array_split(self.route_dict['y'], no_of_segments)
        # hs = np.array_split(self.route_dict['yaw'], no_of_segments)
        # subroute = {}

        # segment size in indices
        index_segment_size = int(self.points_on_route / self.no_of_segments)

        # get starting indices and coords for each segment
        indices = [0]
        staring_coords = []
        coord_dict = {'x': self.route_dict['x'][0],
                      'y': self.route_dict['y'][0],
                      'yaw': self.route_dict['yaw'][0]}
        staring_coords.append(coord_dict)
        for i in range(1, self.no_of_segments-1):
            indices.append(indices[-1] + index_segment_size)
            coord_dict = {'x':self.route_dict['x'][indices[-1]],
                          'y':self.route_dict['y'][indices[-1]],
                          'yaw':self.route_dict['yaw'][indices[-1]]}
            staring_coords.append(coord_dict)
        self.segment_indices = indices
        return indices, staring_coords

    def calc_errors(self, trajectory):
        if self.is_segmented:
            # TODO: for sgement we need to calculate the seq. error but starting 
            # the search at the start of each segment 
            return angular_error(self.route_dict, trajectory)
        else:
            return seq_angular_error(self.route_dict, trajectory)

    def min_dist_from_route(self, xy, start=0, stop=None):
        dist = cdist([xy], np.column_stack((self.route_dict['x'][start:stop], self.route_dict['y'][start:stop])), 'euclidean')
        min_dist = np.min(dist)
        min_idx = np.argmin(dist) + start
        min_xy = (self.route_dict['x'][min_idx], self.route_dict['y'][min_idx])
        return min_idx, min_dist, min_xy

    def dist_from_route_end(self, xy):
        dist = cdist([xy], np.column_stack((self.route_dict['x'][-1], self.route_dict['y'][-1])), 'euclidean')
        return dist.item()
    
    def dist_from_start(self, xy):
        dx = xy[0] - self.route_dict['x'][0]
        dy = xy[1] - self.route_dict['y'][0]
        return np.sqrt(dx**2+dy**2)

    def get_tavel_distance(self):
        return travel_dist(self.route_dict['x'], self.route_dict['y'])

    def get_mean_curv(self):
        return meancurv2d(self.route_dict['x'], self.route_dict['y'])

    def get_route_dict(self):
        return self.route_dict

    def get_imgs(self):
        return self.route_dict['imgs']

    def get_qimgs(self):
        return self.route_dict['qimgs']
    
    def get_qyaw(self): 
        return self.route_dict['qyaw']

    def get_qxycoords(self):
        return {'x': self.route_dict['qx'], 'y': self.route_dict['qy']}

    def get_xycoords(self):
        return {'x': self.route_dict['x'], 'y': self.route_dict['y']}
    
    def get_yaw(self): return self.route_dict['yaw']

    def get_pitch(self): return self.route_dict['pitch']

    def get_roll(self): return self.route_dict['roll']

    def get_starting_coords(self):
        return {'x': self.route_dict['x'][0],
                'y': self.route_dict['y'][0],
                'yaw': self.route_dict['yaw'][0]}
    
    def get_route_id(self):
        return self.route_id

def load_routes(path, ids, **kwargs):
    routes = []
    for id in ids:
        route_path =  os.path.join(path, 'route{}'.format(id))
        r = Route(route_path, id, **kwargs)
        routes.append(r)
    return routes


class BoBRoute:

    def __init__(self, path, route_id=None, read_imgs=True, unwraper=Unwraper, sample_step=1, **kwargs):
        self.path = path
        self.read_imgs = read_imgs
        self.proc_imgs = []
        self.proc_qimgs= []
        self.route_id = str(route_id)
        self.unwraper = unwraper
        # mDefult resizing to the max size needed for the benchmarks
        self.img_shape = (720, 150)
        self.resizer = resize(self.img_shape)
        # self.vcrop = vcrop
        # # change vcrop from percentage to an actual row index
        # self.vcrop = int(round(self.img_shape[1] * self.vcrop))
        self.sample_step = sample_step

        self.route_dict = self.load_route()


    def load_route(self):
        route_data = pd.read_csv(os.path.join(self.path, 'database_entries.csv'), index_col=False)
        route_data = route_data[route_data["X [mm]"].notnull()]
        route_data.rename(str.strip, axis='columns', inplace=True, errors="raise")
        # rename columns to filename,pitch,roll,x,y,yaw,z
        route_data = route_data.to_dict('list')
        key_maps = {'X [mm]': 'x', 'Y [mm]': 'y', 'Z [mm]':'z', 
                    'Heading [degrees]':'yaw', 
                    'IMU pitch [degrees]':'pitch', 
                    'IMU roll [degrees]':'roll',
                    'Filename':'filename'}
        for k in key_maps:
            #check the key exists
            if route_data.get(k):
                route_data[key_maps[k]] = route_data.pop(k)
        # convert the lists to numpy arrays
        for k in route_data:
            route_data[k] = np.array(route_data[k])
        route_data['yaw'] = squash_deg(route_data['yaw'])
        # print(route_data.keys())
        if self.read_imgs:
            imgs = []
            for i in route_data['filename']:
                #careful the filenames contain a leading space
                im_path = os.path.join(self.path, i.strip())
                img = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
                imgs.append(img)
            # unwrap the images
            if self.unwraper:
                self.unwraper = self.unwraper(imgs[0])
                for i, im in enumerate(imgs):
                    im = self.unwraper.unwarp(im)
                    im = self.resizer(im)
                    imgs[i] = im
            route_data['imgs'] = imgs
        return route_data
    
    def calc_errors(self, trajectory):
        r_sample = {'x': self.route_dict['x'][::self.sample_step], 
                'y': self.route_dict['y'][::self.sample_step],
                'yaw': self.route_dict['yaw'][::self.sample_step]
                }
        return seq_angular_error(r_sample, trajectory)

    def set_query_data(self, qx, qy, qyaw, qimgs):
        self.route_dict['qx'] = np.array(qx)
        self.route_dict['qy'] = np.array(qy)
        self.route_dict['qyaw'] = np.array(qyaw)
        self.route_dict['qimgs'] = qimgs

    def get_xycoords(self):
        return {'x': self.route_dict['x'][::self.sample_step], 
                'y': self.route_dict['y'][::self.sample_step],}
    
    def get_qxycoords(self):
        return {'x': self.route_dict['qx'], 'y': self.route_dict['qy']}
    
    def get_mean_curv(self):
        return meancurv2d(self.route_dict['x'], self.route_dict['y'])

    def get_yaw(self): return self.route_dict['yaw'][::self.sample_step]

    def get_qyaw(self): return self.route_dict['qyaw']

    def get_pitch(self): return self.route_dict['pitch'][::self.sample_step]

    def get_roll(self): return self.route_dict['roll'][::self.sample_step]
    
    def get_imgs(self):
        ' this could be one line but for clarity i am showing the use of the variable sample_step'
        if self.sample_step > 1:
            return self.route_dict['imgs'][::self.sample_step]
        else:
            return self.route_dict['imgs']

    def get_qimgs(self):
        return self.route_dict['qimgs']

    def get_route_dict(self):
        return self.route_dict
    
    def get_route_id(self):
        return self.route_id
    
def load_bob_routes(path, ids, suffix=None, repeats=None, **kwargs):
    routes = []
    # Thiis the the reference route choosen from the repeats. Usualy the first one.
    ref_route_repeat_id = 1
    for id in ids:
        route_path =  os.path.join(path, 'route{}'.format(id))
        if suffix:
            route_path = os.path.join(route_path, suffix)
        # the referencee route is always 0, i.e the first route recorded
        route_path = route_path + str(ref_route_repeat_id)
        r = BoBRoute(route_path, route_id=id, **kwargs)
        if repeats:
            repeats_path =  os.path.join(path, 'route{}'.format(id))
            make_query_repeat_routes(r, ref_route_repeat_id, repeats_path, repeats,
                                     suffix=suffix, **kwargs)
        routes.append(r)
    return routes

def make_query_repeat_routes(route, route_ref_id, rep_path, repeats, suffix=None, **kwargs):
    repeats = [*range(1, repeats+1)]
    repeats.remove(route_ref_id)
    if suffix:
        rep_path = os.path.join(rep_path, suffix)
    qx = []
    qy = []
    qyaw = []
    qimgs = []
    for rep in repeats:
        route_path = rep_path + str(rep)
        r = BoBRoute(route_path, **kwargs)
        xy = r.get_xycoords()
        qx.extend(xy['x'])
        qy.extend(xy['y'])
        qyaw.extend(r.get_yaw())
        qimgs.extend(r.get_imgs())
    route.set_query_data(qx, qy, qyaw, qimgs)


def load_bob_routes_repeats(path, ids, suffix=None, ref_route=1, repeats=None, **kwargs):
    routes = []
    repeat_routes = []
    # Thiis the the reference route choosen from the repeats. Usualy the first one.
    ref_route_id = ref_route
    for rid in ids:
        route_path =  os.path.join(path, 'route{}'.format(rid))
        if suffix:
            route_path = os.path.join(route_path, suffix)
            #this is the preat routes path with the suffix
            repeats_path = route_path
        # the referencee route is always 0, i.e the first route recorded
        route_path = route_path + str(ref_route_id)
        r = BoBRoute(route_path, route_id=rid, **kwargs)
        routes.append(r)
        if repeats:
            repeat_ids = [*range(1, repeats+1)]
            repeat_ids.remove(ref_route_id)
            rep_routes_temp_l = []
            for rep in repeat_ids:
                route_path = repeats_path + str(rep)
                r = BoBRoute(route_path, route_id=rep, **kwargs)
                rep_routes_temp_l.append(r)
            repeat_routes.append(rep_routes_temp_l)
    return routes, repeat_routes

import os
import antworld
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from source.imgproc import Pipeline
from source.utils import check_for_dir_and_create, write_route, squash_deg, travel_dist, pol2cart
from source.gencoords import generate_from_points, generate_grid

# Old Seville data (lower res, but loads faster)
worldpath = antworld.bob_robotics_path + "/resources/antworld/world5000_gray.bin"
# z = 0.01 # m

# New Seville data
worldpath = antworld.bob_robotics_path + "/resources/antworld/seville_vegetation_downsampled.obj"
# print(antworld.bob_robotics_path)
z = 1.5 # m (for some reason the ground is at ~1.5m for this world)


class Agent:
    def __init__(self, pitch_roll_sig=None, repos_thresh=None):
        self.agent = antworld.Agent(720, 150)
        (xlim, ylim, zlim) = self.agent.load_world(worldpath)
        print(xlim, ylim, zlim)
        self.pitch_roll_noise = pitch_roll_sig
        #Using self.z means that the height is constant
        self.z = z
        # holds current xy possition
        self.xy = None
        # hold current heading
        self.h = None
        # if the sigma of the gaussian noise is provided use a noise function for pitch and roll noise
        if pitch_roll_sig:
            self.noise = lambda : np.random.normal(scale=pitch_roll_sig)
        else:
            self.noise = lambda : 0
        # trial_fail_count (TFC) params
        self.repos_thresh = repos_thresh
        # the size of the reposition window search for closest point
        self.repos_w = 30
        self.trial_fail_count = None
        self.tfc_indices = []

        self.pipe = Pipeline()
        # the route object
        self.route = None
        # the navigator object
        self.nav = None
        # keep track of the distance from the start of the route
        self.prev_dist = 0
        # keep track of the index 
        self.prev_idx = 0
        # keeps track of the curretn trajectory
        self.traj = {'x': [], 'y': [], 'heading': []}

    def set_seed(self, seed):
        np.random.seed(seed)

    def record_route(self, route, path, route_id=1):
        check_for_dir_and_create(path)
        x = route['x']
        y = route['y']
        z = route['z']
        headings = route['yaw']
        # Fixed high for now
        # TODO: in the future it may be a good idea to adap the coed to use the elevation
        #   and the pitch, roll noise
        route['filename'] = []

        for i, (xi, yi, h1) in enumerate(zip(x, y, headings)):
            self.agent.set_position(xi, yi, self.z)
            self.agent.set_attitude(h1, 0, 0)
            img = self.agent.read_frame()
            filename = os.path.join(path, "img%i.png" % i )
            # filename = path + "img%i.png" % i
            cv2.imwrite(filename, img)
            route['filename'].append("img%i.png" % i)

        write_route(path, route, route_id=route_id)

    def rec_route_from_points(self, path, route_id=1, generator='circle', **kwargs):
        # Augment the directory containing all route
        # to create a new directory w.r.t the new route
        path = os.path.join(path, 'route' + str(route_id))
        check_for_dir_and_create(path, remove=True)
        # Generate coordinates and write them to file
        route = generate_from_points(path, generator=generator, **kwargs)
        self.record_route(route, path, route_id=route_id)

    def get_img(self, xy, deg):
        '''
        Render a greyscale image from the antworld given an xy position and heading
        :param xy:
        :param deg:
        :return:
        '''
        self.agent.set_position(xy[0], xy[1], self.z)
        self.agent.set_attitude(deg, self.noise(), self.noise())
        img = cv2.cvtColor(self.agent.read_frame(), cv2.COLOR_BGR2GRAY)
        return self.pipe.apply(img)

    def update_position(self, xy, deg, r):
        rad = deg * (np.pi / 180)
        # x and y are inverted because the antworld had the 0 degree origin north
        # this corresponds to the flipping of x, y when calculating the new Cartesian
        # direction
        y, x = pol2cart(r, rad)

        xx = xy[0] + x
        yy = xy[1] + y

        img = self.get_img((xx, yy), deg)

        return (xx, yy), img

    def test_nav(self, coords, r=0.05, t=100, sigma=0.1, **kwargs):
        #TODO: Here we need to initialise the progrees tracking variables. 
        # It is not clear how that would work with the route segmentation
        # keep track of the index progress
        #TODO: prev index should be initialised as the first index
        self.prev_idx = 0

        self.repos_thresh = kwargs.get('repos_thresh')
        # random initial position and heading
        # near the first location of the route
        if sigma:
            self.h = np.random.randint(0, 360)
            x = coords['x']
            x = np.random.normal(x, sigma)
            y = coords['y']
            y = np.random.normal(y, sigma)
            self.xy = (x, y)
        else:
            self.xy = (coords['x'], coords['y'])
            self.h = coords['yaw']


        # Place agent to the initial position and render the image
        img = self.get_img(self.xy, self.h)

        # initialise the log variables
        self.traj = {'x': [], 'y': [], 'heading': []}

        # Navigation loop
        for i in range(0, t):
            self.i = i
            # get the new heading from teh navigator and format it properly
            new_h = self.nav.get_heading(img)
            self.h = self.h + new_h
            self.h = squash_deg(self.h)

            # reposition the agent and get the new image
            self.xy, img = self.update_position(self.xy, self.h, r)
            self.check4reposition()

            img = self.get_img(self.xy, self.h)
            # log the coordinates and attitude
            self.traj['x'].append(self.xy[0])
            self.traj['y'].append(self.xy[1])
            self.traj['heading'].append(self.h)

            # Check for termination conditions
            if self.check4route_end():
                break


        for k in self.traj.keys():
            self.traj[k] = np.array(self.traj[k])
        return self.traj, self.nav

    def check4route_end(self):
        return self.route.dist_from_route_end(self.xy) <= self.repos_thresh
    
    def check4reposition(self):
        if (self.i + 1) % 10 == 0:
            
            # check distance from the closest point on the route
            start = max(self.prev_idx, 0)
            stop = min(self.prev_idx+self.repos_w, self.route.route_end)
            idx, dist, xy = self.route.min_dist_from_route(self.xy, start=start, stop=stop)
            # glob_idx, _, _ = self.route.min_dist_from_route(self.xy)
            # print(f'({start}, {stop})')
            # print('simul. i:', self.i, ' prev_idx:', self.prev_idx, ' current best i: ', idx, ' glob_idx: ', glob_idx, ' dist: ', dist)
            if dist >= self.repos_thresh:
                #print('dist repos!')
                self.reposition(idx=idx)
                return
            
            # check progress of the index closest to the query point
            if idx <= self.prev_idx:
                #print('index repos!')
                self.reposition(idx=idx)
                return
            self.prev_idx = idx
        # check distance form the start of the route
        # dist = self.route.dist_from_start(self.xy)
        # if (self.i + 1) % 10 == 0:
        #     if dist <= self.prev_dist:
        #         self.xy = xy
        #         self.trial_fail_count += 1
        #         self.nav.reset_window(idx)
        #     self.prev_dist = dist
    def reposition(self, idx: int, idx_offset=5):
        '''
        Reposition the agent by index.
        Using the offset by default adds 5 to the index
        '''
        # import pdb; pdb.set_trace()
        idx += idx_offset
        self.prev_idx = idx
        # check you are not near the end of the route already
        if idx >= self.route.route_end: idx = self.route.route_end - 1
        coords = self.route.get_xycoords()
        self.xy = (coords['x'][idx].item(), coords['y'][idx].item())
        self.h = self.route.get_yaw()[idx].item()
        self.trial_fail_count += 1
        self.nav.reset_window(idx)
        self.tfc_indices.append(self.i)


    def segment_test(self, route, nav, segment_length=3, **kwargs):
        #TODO: The progress tracking variable should be handled appopriately here.
        trajectories = {'x': [], 'y': [], 'heading': []}
        # get starting indices for each segment
        indices, starting_coords = route.segment_route(segment_length)
        for i, coord in enumerate(starting_coords):
            self.nav.reset_window(indices[i])
            traj, nav = self.test_nav(coord, **kwargs)
            # Append the segment trajectory to the log
            for k in trajectories:
                trajectories[k] = np.append(trajectories[k], traj[k])

        return trajectories, nav

    def run_agent(self, route, nav, segment_length=None, **kwargs):
        self.trial_fail_count = 0
        self.tfc_indices = []
        self.route = route
        self.pipe = Pipeline(**kwargs)
        self.nav = nav
        if segment_length:
            return self.segment_test(route, segment_length, **kwargs)
        else:
            coords = self.route.get_starting_coords()
            return self.test_nav(coords, **kwargs)

    def rec_grid(self, steps, path):
        path = path + 'grid' + str(steps) + '/'
        grid = generate_grid(steps)
        self.record_route(grid, path)

    def get_trial_fail_count(self):
        return self.trial_fail_count
    
    def get_tfc_indices(self):
        return self.tfc_indices


"""
Testing
"""
# agent = Agent()
# # # rec_grid(70, path='../new-antworld/')
# route_id = 3
# path = '../new-antworld/exp1/'
# path = '../test_data/'
# # agent.rec_route_from_points(path, route_id=route_id, generator='line', start=-2, end=2, sigma=0.2, curve_points=100)
# agent.rec_route_from_points(path, route_id=route_id, generator='gauss', mean=(0, 0), sigma=3, curve_points=100)

#
# record_route(datapoints, "../new-antworld/route2/")
# agent.set_position(0, 0, z)
# agent.set_attitude(0, 0, 0)
#
# xy, img = update_position((0, 0), 45, 0.5)
#
# print(xy, img.shape)
#



# pitch = 0.0
# roll = 0.0
# x=0
# y=0
#
# deg = [0, 90, 180, 270, 360]
# # deg = [0, -90, -180, -270, -360]
#
#
# for i, yaw in enumerate(deg):
#     agent.set_position(x, y, z)
#     agent.set_attitude(yaw, pitch, roll)
#     im = agent.read_frame()
#     filename = "test_data/antworld%i.png" % yaw
#     cv2.imwrite(filename, im)

#
# ys = np.arange(0, 2, 0.1)
# yaw = 0
# imgid = 0
# for i in ys:
#     agent.set_position(x, y + i, z)
#     agent.set_attitude(yaw, pitch, roll)
#     im = agent.read_frame()
#     filename = "test_data/ys%i.png" % imgid
#     cv2.imwrite(filename, im)
#     imgid += 1
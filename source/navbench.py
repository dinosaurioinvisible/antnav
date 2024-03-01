import os
import copy
import pandas as pd
import time
import itertools
import multiprocessing
import functools
import numpy as np
import yaml
from source.utils import pre_process, load_route_naw, check_for_dir_and_create, calc_dists, squash_deg
from source import seqnav as spm, perfect_memory as pm
from source.routedatabase import Route, load_routes, load_bob_routes, load_bob_routes_repeats
from source.imgproc import Pipeline
from source import infomax


class Benchmark:
    def __init__(self, results_path, routes_path, grid_path=None,  filename='results.csv', 
                 route_path_suffix=None, grid_dist=None, route_repeats=None, bench_data=None):
        self.results_path = results_path
        self.routes_path = routes_path
        self.route_path_suffix = route_path_suffix
        self.route_repeats = route_repeats
        self.grid_path = grid_path
        check_for_dir_and_create(results_path)
        self.jobs = 0
        self.total_jobs = 1
        self.bench_logs = []
        self.route_ids = None
        self.routes_data = []
        #self.dist = 0.2  # Distance between grid images and route images
        self.dist = grid_dist
        self.bench_data = bench_data
        self.log = {'route_id': [], 'blur': [], 'edge': [], 'res': [], 'window': [],
                    'matcher': [], 'mean_error': [], 'errors': [], 'seconds': [],
                    'abs_index_diff': [], 'window_log': [], 'best_sims': [], 'dist_diff': [],
                    'tx': [], 'ty': [], 'th': []}

    def load_routes(self, route_ids):
        self.route_ids = route_ids
        for rid in self.route_ids:
            route_path = self.routes_path + '/route' + str(rid) + '/'
            route_data = load_route_naw(route_path, route_id=id, query=True,  max_dist=self.dist)
            self.routes_data.append(route_data)

    def get_grid_chunks(self, grid_gen, chunks=1):
        lst = list(grid_gen)
        return [lst[i::chunks] for i in range(chunks)]

    def remove_blur_edge(self, combo):
        # removes a combination if it has both blurring and edges
        return not (combo['edge_range'] and combo['blur'])

    def remove_non_blur_edge(self, combo):
        return not combo.get('edge_range') and not combo.get('blur') and not combo.get('gauss_loc_norm') and not combo.get('loc_norm')
        #return not combo['edge_range'] and not combo['blur']

    def remove_edge_loc_gloc_combos(self, combo):
        return not (combo.get('edge_range') and combo.get('loc_norm') 
                    and combo.get('gauss_loc_norm'))
    
    def remove_edge_gloc_combos(self, combo):
        return not (combo.get('edge_range') and combo.get('gauss_loc_norm'))  
    
    def remove_edge_loc_combos(self, combo):
        return not (combo.get('edge_range') and combo.get('loc_norm'))  

    def remove_gloc_loc_combos(self, combo):
        return not (combo.get('gauss_loc_norm') and combo.get('loc_norm')) 

    def get_grid_dict(self, params):
        grid = itertools.product(*[params[k] for k in params])

        grid = [*grid]
        grid_dict = []
        for combo in grid:
            combo_dict = {}
            for i, k in enumerate(params):
                combo_dict[k] = combo[i]
            grid_dict.append(combo_dict)

        if params.get('edge_range') or params.get('loc_norm') or params.get('gauss_loc_norm'):
            #grid_dict[:] = [x for x in grid_dict if self.remove_blur_edge(x)]
            #grid_dict[:] = [x for x in grid_dict if not self.remove_non_blur_edge(x)]
            grid_dict[:] = [x for x in grid_dict if self.remove_edge_loc_gloc_combos(x)]
            grid_dict[:] = [x for x in grid_dict if self.remove_edge_gloc_combos(x)]
            grid_dict[:] = [x for x in grid_dict if self.remove_edge_loc_combos(x)]
            grid_dict[:] = [x for x in grid_dict if self.remove_gloc_loc_combos(x)]
        return grid_dict

    @staticmethod
    def _init_shared():
        manager = multiprocessing.Manager()
        # shared = manager.list([0, 0])
        shared = manager.dict({'jobs': 0, 'total_jobs': 0})
        return shared

    def bench_paral(self, params, route_ids=None, cores=None):
        # save the parmeters of the test in a json file
        check_for_dir_and_create(self.results_path)
        param_path = os.path.join(self.results_path, 'params.yml')
        temp_params = copy.deepcopy(params)
        temp_params['routes_path'] = self.routes_path
        temp_params['route_ids'] = route_ids
        with open(param_path, 'w') as fp:
            yaml.dump(temp_params, fp)

        existing_cores = multiprocessing.cpu_count()
        if cores and cores > existing_cores:
            cores = existing_cores - 1
        elif cores and cores <= existing_cores:
            cores = cores
        else:
            cores = existing_cores - 1
        print(existing_cores, ' CPU cores found. Using ', cores, ' cores')

        grid = self.get_grid_dict(params)
        shared = self._init_shared()
        self.total_jobs = len(grid)
        shared['total_jobs'] = self.total_jobs * len(route_ids)

        if self.total_jobs < cores:
            no_of_chunks = self.total_jobs
        else:
            no_of_chunks = cores
        # Generate list chunks of grid combinations
        chunks = self.get_grid_chunks(grid, no_of_chunks)

        print('{} combinations, testing on {} routes, running on {} cores'.format(self.total_jobs, len(route_ids), no_of_chunks))
        
        #Make a dict of parameter arguments
        arg_params = {'route_ids': route_ids, 'grid_dist':self.dist,
                      'routes_path':self.routes_path, 'grid_path':self.grid_path,
                      'route_path_suffix':self.route_path_suffix,
                      'repeats':self.route_repeats, 'results_path':self.results_path}
        # Partial callable
        #TODO: here i need to decide on a worked based on the dataset.
        if self.bench_data == 'ftl':
            worker = functools.partial(self.worker_bench_repeats, arg_params, shared)
        elif self.bench_data == 'aw2':
            worker = functools.partial(self.worker_bench, arg_params, shared)
        else:
            raise Exception("Provide database type")

        pool = multiprocessing.Pool(processes=no_of_chunks)

        logs = pool.map_async(worker, chunks)
        pool.close()
        pool.join()

        return logs

    def bench_singe_core(self, params, route_ids=None):
        self._total_jobs(params)

        # Get list of parameter keys
        keys = [*params]

        grid = itertools.product(*[params[k] for k in params])

        #  Go though all combinations in the grid
        for combo in grid:

            # create combo dictionary
            combo_dict = {}
            for i, k in enumerate(keys):
                combo_dict[k] = combo[i]

            matcher = combo_dict['matcher']
            window = combo_dict['window']
            window_log = None
            path = '../new-antworld/'
            for route_id in route_ids:  # for every route
                # TODO: In the future the code below (inside the loop) should all be moved inside the navigator class
                route_path = '../new-antworld/route' + str(route_id) + '/'
                # route = load_route_naw(route_path, route_id=route_id, imgs=True, query=True, max_dist=0.2)
                # _, test_x, test_y, test_imgs, route_x, route_y, \
                #     route_heading, route_imgs = load_route(route, self.dist)
                route = Route(route_path, route_id, grid_path=self.grid_path)
                
                tic = time.perf_counter()
                # Preprocess images
                test_imgs = pre_process(route.get_qimgs(), combo_dict)
                route_imgs = pre_process(route.get_imgs(), combo_dict)
                # Run navigation algorithm
                if window:
                    nav = spm.SequentialPerfectMemory(route_imgs, matcher, window=window)
                    recovered_heading, window_log = nav.navigate(test_imgs)
                else:
                    nav = pm.PerfectMemory(route_imgs, matcher)
                    recovered_heading = nav.navigate(test_imgs)

                toc = time.perf_counter()
                # Get time complexity
                time_compl = toc-tic
                # Get the errors and the minimum distant index of the route memory
                # errors, min_dist_index = degree_error(test_x, test_y, route_x, route_y, route_heading, recovered_heading)
                traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
                errors, min_dist_index = route.calc_errors(traj)
                # Difference between matched index and minimum distance index and distance between points
                matched_index = nav.get_index_log()
                abs_index_diffs = np.absolute(np.subtract(nav.get_index_log(), min_dist_index))
                dist_diff = calc_dists(route.get_xycoords(), min_dist_index, matched_index)
                mean_route_error = np.mean(errors)
                self.log['route_id'].extend([route_id])
                self.log['blur'].extend([combo_dict.get('blur')])
                self.log['edge'].extend([combo_dict.get('edge_range')])
                self.log['res'].append(combo_dict.get('shape'))
                self.log['window'].extend([window])
                self.log['matcher'].extend([matcher])
                self.log['mean_error'].append(mean_route_error)
                self.log['seconds'].append(time_compl)
                self.log['window_log'].append(window_log)
                self.log['best_sims'].append(nav.get_best_sims())
                self.log['tx'].append(traj['x'].tolist())
                self.log['ty'].append(traj['y'].tolist())
                self.log['th'].append(traj['heading'].tolist())
                self.log['abs_index_diff'].append(abs_index_diffs.tolist())
                self.log['dist_diff'].append(dist_diff.tolist())
                self.log['errors'].append(errors)
            self.jobs += 1
            print('Jobs completed: {}/{}'.format(self.jobs, self.total_jobs))
        return self.log

    def _total_jobs(self, params):
        for k in params:
            self.total_jobs = self.total_jobs * len(params[k])
        print('Total number of jobs: {}'.format(self.total_jobs))

    def benchmark(self, params, route_ids, parallel=True, cores=None):
        print(f'testing params: {params}')
        assert isinstance(params, dict)
        assert isinstance(route_ids, list)

        if parallel:
            self.log = None
            self.log = self.bench_paral(params, route_ids, cores=cores)
            self.unpack_results()
        else:
            self.log = self.bench_singe_core(params, route_ids)

        bench_results = pd.DataFrame(self.log)
        write_path = os.path.join(self.results_path, 'results.csv')
        bench_results.to_csv(write_path, index=False)
        #print(bench_results)

    def unpack_results(self):
        results = self.log.get()
        print(len(results), 'Results produced')
        self.log = results[0]
        for dictionary in results[1:]:
            for k in dictionary:
                self.log[k].extend(dictionary[k])


    @staticmethod
    def worker_bench(arg_params, shared, chunk):
        # unpack shared bench parameters
        route_ids = arg_params.get('route_ids')
        dist =  arg_params.get('grid_dist')
        routes_path =  arg_params.get('routes_path')
        grid_path =  arg_params.get('grid_path')
        results_path = arg_params.get('results_path')
        chunk_id = multiprocessing.current_process()._identity

        log = {'route_id': [], 'blur': [], 'edge': [], 'res': [],  'histeq':[], 'vcrop':[], 
               'window': [], 'matcher': [], 'deg_range':[], 'mean_error': [], 
               'seconds': [], 'errors': [], 'abs_index_diff': [], 'window_log': [], 
               'matched_index': [], 'dist_diff': [], 'tx': [], 'ty': [], 'th': [],
               'ah': [] , 'rmfs_file':[],'best_sims':[], 'loc_norm':[], 
               'gauss_loc_norm':[], 'wave':[], 'nav-name':[]}
        
        # Load all routes
        routes = load_routes(routes_path, route_ids, max_dist=dist, grid_path=grid_path)

        #  Go though all combinations in the chunk
        for combo in chunk:

            matcher = combo['matcher']
            window = combo['window']
            window_log = None
            for route in routes:  # for every route

                tic = time.perf_counter()
                # Preprocess images
                pipe = Pipeline(**combo)
                route_imgs = pipe.apply(route.get_imgs())
                test_imgs = pipe.apply(route.get_qimgs())
                # Run navigation algorithm
                if window:
                    nav = spm.SequentialPerfectMemory(route_imgs, matcher, **combo)
                    recovered_heading, window_log = nav.navigate(test_imgs)
                elif window == 0:
                    nav = pm.PerfectMemory(route_imgs, matcher, **combo)
                    recovered_heading = nav.navigate(test_imgs)
                else:
                    infomaxParams = infomax.Params()
                    nav = infomax.InfomaxNetwork(infomaxParams, route_imgs, **combo)
                    recovered_heading = nav.navigate(test_imgs)

                toc = time.perf_counter()
                # Get time complexity
                time_compl = toc - tic
                # Get the errors and the minimum distant index of the route memory
                qxy = route.get_qxycoords()
                traj = {'x': qxy['x'], 'y': qxy['y'], 'heading': recovered_heading}
                #!!!!!! Important step to get the heading in the global coord system
                traj['heading'] = squash_deg(route.get_qyaw() + recovered_heading)
                errors, min_dist_index = route.calc_errors(traj)
                # Difference between matched index and minimum distance index and distance between points
                matched_index = nav.get_index_log()
                if matched_index:
                    abs_index_diffs = np.absolute(np.subtract(nav.get_index_log(), min_dist_index))
                    dist_diff = calc_dists(route.get_xycoords(), min_dist_index, matched_index)
                    abs_index_diffs = abs_index_diffs.tolist()
                    dist_diff = dist_diff.tolist()
                else:
                    abs_index_diffs = None
                    dist_diff = None
                
                mean_route_error = np.mean(errors)
                window_log = nav.get_window_log()
                rec_headings = nav.get_rec_headings()
                deg_range = nav.deg_range

                rmf_logs = np.array(nav.get_rsims_log(), dtype=object)
                rmf_logs_file = 'rmfs' + str(chunk_id) + str(shared['jobs'])
                rmfs_path = os.path.join(results_path, rmf_logs_file)
                np.save(rmfs_path, rmf_logs)

                log['nav-name'].append(nav.get_name())
                log['route_id'].append(route.get_route_id())
                log['blur'].append(combo.get('blur'))
                log['histeq'].append(combo.get('histeq'))
                log['edge'].append(combo.get('edge_range'))
                log['res'].append(combo.get('shape'))
                log['vcrop'].append(combo.get('vcrop'))
                log['window'].append(window)
                log['loc_norm'].append(combo.get('loc_norm'))
                log['gauss_loc_norm'].append(combo.get('gauss_loc_norm'))
                log['wave'].append(combo.get('wave'))
                log['matcher'].append(matcher)
                log['deg_range'].append(deg_range)
                log['mean_error'].append(mean_route_error)
                log['seconds'].append(time_compl)
                log['window_log'].append(window_log)
                log['rmfs_file'].append(rmf_logs_file)
                log['tx'].append(traj['x'].tolist())
                log['ty'].append(traj['y'].tolist())
                # This is the heading in the global coord system
                log['th'].append(traj['heading'].tolist())
                # This is the agent heading from the egocentric agent reference
                log['ah'].append(rec_headings)
                log['matched_index'].append(matched_index)
                log['abs_index_diff'].append(abs_index_diffs)
                log['dist_diff'].append(dist_diff)
                log['errors'].append(errors)
                log['best_sims'].append(nav.get_best_sims())
                # Increment the complete jobs shared variable
                shared['jobs'] = shared['jobs'] + 1
                print(multiprocessing.current_process(), ' jobs completed: {}/{}'.format(shared['jobs'], shared['total_jobs']))
        return log
    
    @staticmethod
    def worker_bench_repeats(arg_params, shared, chunk):
        # unpack shared bench parameters
        route_ids = arg_params.get('route_ids')
        dist =  arg_params.get('grid_dist')
        routes_path =  arg_params.get('routes_path')
        grid_path =  arg_params.get('grid_path')
        results_path = arg_params.get('results_path')
        route_path_suffix = arg_params.get('route_path_suffix')
        repeats = arg_params.get('repeats')
        chunk_id = multiprocessing.current_process()._identity

        log = {'route_id': [],'ref_route':[], 'rep_id': [], 'sample_rate':[], 'blur': [], 
               'histeq':[], 'edge': [], 'res': [], 'vcrop':[],'window': [], 'matcher': [],
               'deg_range':[], 'mean_error': [], 'seconds': [], 'errors': [], 
               'abs_index_diff': [], 'window_log': [], 'matched_index': [], 'dist_diff': [], 
               'tx': [], 'ty': [], 'th': [],'ah': [] , 'rmfs_file':[], 'best_sims':[], 
               'loc_norm':[], 'gauss_loc_norm':[], 'wave':[], 'nav-name':[]}

        
        # Load all routes
        # routes = load_routes(routes_path, route_ids, max_dist=dist, grid_path=grid_path)

        #  Go though all combinations in the chunk
        for combo in chunk:
            routes, repeat_routes = load_bob_routes_repeats(routes_path, route_ids, suffix=route_path_suffix, repeats=repeats, **combo)
            matcher = combo.get('matcher')
            window = combo.get('window')
            window_log = None
            for ri, route in enumerate(routes):  # for every route
                
                for rep_route in repeat_routes[ri]: # for every repeat route
                    tic = time.perf_counter()
                    # Preprocess images
                    pipe = Pipeline(**combo)
                    route_imgs = pipe.apply(route.get_imgs())
                    test_imgs = pipe.apply(rep_route.get_imgs())
                    # Run navigation algorithm
                    if window:
                        nav = spm.SequentialPerfectMemory(route_imgs, matcher, **combo)
                        recovered_heading, window_log = nav.navigate(test_imgs)
                    elif window == 0:
                        nav = pm.PerfectMemory(route_imgs, matcher, **combo)
                        recovered_heading = nav.navigate(test_imgs)
                    else:
                        infomaxParams = infomax.Params()
                        nav = infomax.InfomaxNetwork(infomaxParams, route_imgs, **combo)
                        recovered_heading = nav.navigate(test_imgs)
                    # here i need a navigate method for infomax.
                    toc = time.perf_counter()
                    # Get time complexity
                    time_compl = toc - tic
                    # Get the errors and the minimum distant index of the route memory
                    qxy = rep_route.get_xycoords()
                    traj = {'x': qxy['x'], 'y': qxy['y'], 'heading': recovered_heading}
                    #################!!!!!! Important step to get the heading in the global coord system
                    traj['heading'] = squash_deg(rep_route.get_yaw() + recovered_heading)
                    errors, min_dist_index = route.calc_errors(traj)
                    # Difference between matched index and minimum distance index and distance between points
                    matched_index = nav.get_index_log()
                    if matched_index:
                        abs_index_diffs = np.absolute(np.subtract(nav.get_index_log(), min_dist_index))
                        dist_diff = calc_dists(route.get_xycoords(), min_dist_index, matched_index)
                        abs_index_diffs = abs_index_diffs.tolist()
                        dist_diff = dist_diff.tolist()
                    else:
                        abs_index_diffs = None
                        dist_diff = None
                    mean_route_error = np.mean(errors)
                    window_log = nav.get_window_log()
                    rec_headings = nav.get_rec_headings()
                    deg_range = nav.deg_range

                    rmf_logs = np.array(nav.get_rsims_log(), dtype=object)
                    rmf_logs_file = f"rmfs-{chunk_id}-{shared['jobs']}-{rep_route.get_route_id()}"
                    rmfs_path = os.path.join(results_path, rmf_logs_file)
                    np.save(rmfs_path, rmf_logs)

                    log['nav-name'].append(nav.get_name())
                    log['route_id'].append(route.get_route_id())
                    log['ref_route'].append(combo.get('ref_route'))
                    log['rep_id'].append(rep_route.get_route_id())
                    log['sample_rate'].append(combo.get('sample_step'))
                    log['blur'].append(combo.get('blur'))
                    log['histeq'].append(combo.get('histeq'))
                    log['edge'].append(combo.get('edge_range'))
                    log['res'].append(combo.get('shape'))
                    log['vcrop'].append(combo.get('vcrop'))
                    log['window'].append(window)
                    log['loc_norm'].append(combo.get('loc_norm'))
                    log['gauss_loc_norm'].append(combo.get('gauss_loc_norm'))
                    log['wave'].append(combo.get('wave'))
                    log['matcher'].append(matcher)
                    log['deg_range'].append(deg_range)
                    log['mean_error'].append(mean_route_error)
                    log['seconds'].append(time_compl)
                    log['window_log'].append(window_log)
                    log['tx'].append(traj['x'].tolist())
                    log['ty'].append(traj['y'].tolist())
                    # This is the heading in the global coord system
                    log['th'].append(traj['heading'].tolist())
                    # This is the agent heading from the egocentric agent reference
                    log['ah'].append(rec_headings)
                    log['rmfs_file'].append(rmf_logs_file)
                    log['matched_index'].append(matched_index)
                    log['abs_index_diff'].append(abs_index_diffs)
                    log['dist_diff'].append(dist_diff)
                    log['errors'].append(errors)
                    log['best_sims'].append(nav.get_best_sims())
                # Increment the complete jobs shared variable
                shared['jobs'] = shared['jobs'] + 1
                print(multiprocessing.current_process(), ' jobs completed: {}/{}'.format(shared['jobs'], shared['total_jobs']))
        return log

from source import utils
from source.utils import pre_process, calc_dists, load_route_naw, seq_angular_error, check_for_dir_and_create
from source import seqnav as spm, perfect_memory as pm
from source.routedatabase import Route
import time
import itertools
import os
import copy
import pandas as pd
import numpy as np
from source import antworld2 as aw
import pickle
from subprocess import Popen
from queue import Queue, Empty
from threading import Thread
import sys
import yaml


def get_grid_dict(params):
    grid = itertools.product(*[params[k] for k in params])
    grid_dict = []
    for combo in grid:
        combo_dict = {}
        for i, k in enumerate(params):
            combo_dict[k] = combo[i]
        grid_dict.append(combo_dict)
    
    # remove the combination where we have both blur and adge detection
    grid_dict[:] = [x for x in grid_dict if remove_blur_edge(x)]
    # Remove combinations where we have neither blur nor edge detection nor gauss_loc_norm
    # generally if all the main pre-proc functions are not being used then we don't want to test those combinations
    grid_dict[:] = [x for x in grid_dict if not remove_non_blur_edge(x)]

    return grid_dict


def remove_blur_edge(combo):
    return not (combo.get('edge_range') and combo.get('blur'))


def remove_non_blur_edge(combo):
    return not combo.get('edge_range') and not combo.get('blur') and not combo.get('gauss_loc_norm') and not combo.get('loc_norm')


def bench(params, routes_path, route_ids):
    log = {'route_id': [], 'blur': [], 'edge': [], 'res': [], 'window': [],
           'matcher': [], 'mean_error': [], 'seconds': [], 'errors': [],
           'dist_diff': [], 'abs_index_diff': [], 'window_log': [],
           'tx': [], 'ty': [], 'th': []}
    agent = aw.Agent()

    grid = get_grid_dict(params)
    total_jobs = len(grid) * len(route_ids)
    jobs = 0
    #  Go though all combinations in the chunk
    for combo in grid:

        matcher = combo['matcher']
        window = combo['window']
        t = combo['t']
        r = combo['r']
        segment_length = combo.get('segment_l')
        window_log = None
        for route_id in route_ids:  # for every route
            route_path = routes_path + '/route' + str(route_id) + '/'
            route = Route(route_path, route_id)

            tic = time.perf_counter()
            # Preprocess images
            route_imgs = pre_process(route.get_imgs(), combo)
            # Run navigation algorithm
            if window:
                nav = spm.SequentialPerfectMemory(route_imgs, matcher, deg_range=(-180, 180), window=window)
            else:
                nav = pm.PerfectMemory(route_imgs, matcher, deg_range=(-180, 180))

            if segment_length:
                traj, nav = agent.segment_test(route, nav, segment_length=segment_length, t=t, r=r, sigma=None, preproc=combo)
            else:
                start = route.get_starting_coords()
                traj, nav = agent.test_nav(start, nav, t=t, r=r, preproc=combo)

            # agent.run_agent(route, nav, t=t, r=r, preproc=combo)

            toc = time.perf_counter()
            time_compl = toc - tic
            # Get the errors and the minimum distant index of the route memory
            errors, min_dist_index = route.calc_errors(traj)
            # Difference between matched index and minimum distance index and distance between points
            matched_index = nav.get_index_log()
            abs_index_diffs = np.absolute(np.subtract(nav.get_index_log(), min_dist_index))
            dist_diff = calc_dists(route.get_xycoords(), min_dist_index, matched_index)
            mean_route_error = np.mean(errors)
            log['route_id'].extend([route_id])
            log['blur'].extend([combo.get('blur')])
            log['edge'].extend([combo.get('edge_range')])
            log['res'].append(combo.get('shape'))
            log['window'].extend([window])
            log['matcher'].extend([matcher])
            log['mean_error'].append(mean_route_error)
            log['seconds'].append(time_compl)
            log['window_log'].append(window_log)
            log['tx'].append(traj['x'].tolist())
            log['ty'].append(traj['y'].tolist())
            log['th'].append(traj['heading'].tolist())
            log['abs_index_diff'].append(abs_index_diffs.tolist())
            log['dist_diff'].append(dist_diff.tolist())
            log['errors'].append(errors)

            # Increment the complete jobs shared variable
            jobs += 1
            print('jobs completed: {}/{}'.format(jobs, total_jobs))
    return log


def benchmark(results_path, routes_path, params, route_ids,  parallel=False, cores=None, num_of_repeats=None):

    assert isinstance(params, dict)
    assert isinstance(route_ids, list)

    if parallel:
        bench_paral(results_path, params, routes_path, route_ids, cores, 
                    num_of_repeats=num_of_repeats)
        # log = unpack_results(log)
    else:
        log = bench(params, routes_path, route_ids)
        bench_results = pd.DataFrame(log)
        bench_results.to_csv(results_path, index=False)


def _total_jobs(params):
    total_jobs = 1
    for k in params:
        total_jobs = total_jobs * len(params[k])
    print('Total number of jobs: {}'.format(total_jobs))
    return total_jobs


def get_grid_chunks(grid_gen, chunks=1):
    lst = list(grid_gen)
    return [lst[i::chunks] for i in range(chunks)]


def unpack_results(results):
    results = results.get()
    print(len(results), 'Results produced')
    log = results[0]
    for dictionary in results[1:]:
        for k in dictionary:
            log[k].extend(dictionary[k])
    return log


def bench_paral(results_path, params, routes_path, route_ids=None, cores=None, num_of_repeats=None):
    # save the parmeters of the test in a json file
    check_for_dir_and_create(results_path)
    param_path = os.path.join(results_path, 'params.yml')
    temp_params = copy.deepcopy(params)
    temp_params['routes_path'] = routes_path
    temp_params['route_ids'] = route_ids
    # if there are no repeats then only one repeat per route
    if not num_of_repeats:  
        num_of_repeats = 1
    temp_params['num_of_repeats'] = num_of_repeats
    #save params to yaml
    with open(param_path, 'w') as fp:
        yaml.dump(temp_params, fp)

    existing_cores = os.cpu_count()
    if cores and cores > existing_cores:
        cores = existing_cores - 1
    elif cores and cores <= existing_cores:
        cores = cores
    else:
        cores = existing_cores - 1
    print(existing_cores, ' CPU cores found. Using ', cores, ' cores')

    grid = get_grid_dict(params)
    total_jobs = len(grid)

    if total_jobs < cores:
        no_of_chunks = total_jobs
    else:
        no_of_chunks = cores
    # Generate a list of chunks of grid combinations
    chunks = get_grid_chunks(grid, no_of_chunks)
    print('{} combinations, testing on {} routes, running on {} cores'.format(total_jobs, len(route_ids), no_of_chunks))
    chunks_path = 'chunks'
    check_for_dir_and_create(chunks_path, remove=True)
    print('Saving chunks in', chunks_path)

    # Pickle the parameter object to use in the worker script
    for i, chunk in enumerate(chunks):
        params = {'chunk': chunk, 'route_ids': route_ids, 
                'routes_path': routes_path, 
                'results_path':results_path, 
                'i': i,
                'num_of_repeats':num_of_repeats}
        with open('chunks/chunk{}.p'.format(i), 'wb') as file:
            pickle.dump(params, file)
    print('{} chunks pickled'.format(no_of_chunks))


    work_path = os.path.join(os.path.dirname(__file__), 'workerscript.py')
    work_path = 'workerscript.py'
    processes = []
    for i, chunk in enumerate(chunks):
        cmd_list = ['python3', work_path, 'chunks/chunk{}.p'.format(i)]
        p = Popen(cmd_list)
        processes.append(p)

    for p in processes:
        p.wait()

    # combine the results that each worker produces into one .csv file
    combine_results(results_path)

def combine_results(path):
    files = find_csv_filenames(path)
    r = []
    for f in files:
        filepath = os.path.join(path, f)
        r.append(pd.read_csv(filepath)) 
    results = pd.concat(r, ignore_index=True)
    path = os.path.join(path, 'results.csv')
    results.to_csv(path, index=False)

def find_csv_filenames(path_to_dir, suffix=".csv" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


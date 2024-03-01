from utils import load_route, pre_process
import seqnav as spm
import perfect_memory as pm
import timeit
import pandas as pd

route_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
matcher = 'rmse'
pre_proc = {'blur': True, 'shape': (180, 50)}
window = 17  # Best performance window
dist = 100
performance_sheet = pd.DataFrame()

time_complexity = []
for id in route_ids:  # for every route
    _, _, _, world_grid_imgs, _, _, \
    _, route_images = load_route(route_id=id, grid_pos_limit=dist)

    pre_world_grid_imgs = pre_process(world_grid_imgs, pre_proc)
    pre_route_images = pre_process(route_images, pre_proc)

    nav = spm.SequentialPerfectMemory(pre_route_images, matcher, window=window)
    # Time the navigator. Start timer
    tic = timeit.default_timer()
    nav.navigate(pre_world_grid_imgs)
    toc = timeit.default_timer()
    # End of timing
    time_complexity.append(toc-tic)

performance_sheet['spm'] = time_complexity


time_complexity = []
for id in route_ids:  # for every route
    _, _, _, world_grid_imgs, _, _, \
    _, route_images = load_route(route_id=id, grid_pos_limit=dist)

    pre_world_grid_imgs = pre_process(world_grid_imgs, pre_proc)
    pre_route_images = pre_process(route_images, pre_proc)

    nav = pm.PerfectMemory(pre_route_images, matcher)
    # Time the navigator. Start timer
    tic = timeit.default_timer()
    nav.navigate(pre_world_grid_imgs)
    toc = timeit.default_timer()
    # End of timing
    time_complexity.append(toc-tic)
performance_sheet['pm'] = time_complexity
performance_sheet.to_csv('Results/time_complexities.csv')

import os
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader
import numpy as np


def mk_loaders(dataset='nmnist',
               both_sets=True,
               batch_size=128, 
               shuffle_train=True, shuffle_test=True, 
               twindow=1000, tfilter=10000,
               tjitter=False, treversal=False, mergep=False, scramble=False):
    
    dirpath = os.path.abspath(os.path.join(os.getcwd(),'..','{}_data'.format(dataset)))

    sensor_size = tonic.datasets.NMNIST.sensor_size

    txs = []
    dir_txs = 'txs_'
    if tfilter > 0:
        denoise_tx = transforms.Denoise(filter_time = tfilter)
        dir_txs += 'd'
        txs.append(denoise_tx)
    if tjitter:
        time_jitter_tx = transforms.TimeJitter(std = 100, clip_negative=True)
        dir_txs += 'j'
        txs.append(time_jitter_tx)
    if treversal:
        rt_rev_tx = transforms.RandomTimeReversal(p = 1, flip_polarities=False)
        dir_txs += 'r'
        txs.append(rt_rev_tx)
    if mergep: 
        merge_pols_tx = transforms.MergePolarities()
        sensor_size = (34,34,1)
        dir_txs += 'm'
        txs.append(merge_pols_tx)
    if scramble:
        scramble_tx = SaccadeScramble()
        dir_txs += 'x'
        txs.append(scramble_tx)
    if twindow > 0:
        frame_tx = transforms.ToFrame(sensor_size=sensor_size, time_window=twindow)
        dir_txs += 'w'
        txs.append(frame_tx)
    
    frame_transform = transforms.Compose(txs)
    tx_path = os.path.join(dirpath,dir_txs)

    trainset = tonic.datasets.NMNIST(save_to=tx_path, transform=frame_transform, train=True)
    if both_sets:
        testset = tonic.datasets.NMNIST(save_to=tx_path, transform=frame_transform, train=False)
    else:
        transform_test = transforms.Compose([transforms.Denoise(filter_time=10000),
                                             transforms.ToFrame(sensor_size=sensor_size,
                                                                time_window=1000)])
        testset = tonic.datasets.NMNIST(save_to=tx_path, transform=transform_test, train=False)

    cache_path_train = os.path.join(tx_path,'cache/train')
    cached_trainset = DiskCachedDataset(trainset, cache_path=cache_path_train)

    cache_path_test = os.path.join(tx_path,'cache/test')
    cached_testset = DiskCachedDataset(testset, cache_path=cache_path_test)

    trainloader = DataLoader(cached_trainset,
                            batch_size=batch_size,
                            collate_fn=tonic.collation.PadTensors(batch_first=False),
                            shuffle=shuffle_train)

    testloader = DataLoader(cached_testset,
                            batch_size=batch_size,
                            collate_fn=tonic.collation.PadTensors(batch_first=False),
                            shuffle=shuffle_test)

    return trainloader, testloader



class EventScramble:
    def __call__(self, events):
        events = events.copy()
        pols = [0,1]
        for pol in pols:
            locs = np.where(events['p']==pol)[0]
            rlocs = np.copy(locs)
            np.random.shuffle(rlocs)
            events['x'][locs] = events['x'][rlocs]
            events['y'][locs] = events['y'][rlocs]
        return events

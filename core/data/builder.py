import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.registery import DATASET_REGISTRY
from .dataload import prepare_dataset
import gc

from memory_profiler import profile

class DataSet:
    def __init__(self, train_input, train_label):
        self.X = train_input
        self.t = train_label

    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        return self.X[index], self.t[index]   


def build_dataset(cfg, prefix):

    dataset_cfg = copy.deepcopy(cfg)
    try:
        dataset_cfg = dataset_cfg[prefix]
    except Exception:
        raise f'should contain {prefix}!'

    datalist, annlist, weight  = prepare_dataset(dataset_cfg['args']['data_root'], dataset_cfg['args']['label_root'], dataset_cfg['args']['frame_len'], prefix)
    
    dataset = DataSet(datalist, annlist)
    
    del datalist
    del annlist
    # del input
    # del ann
    gc.collect()
    # print(np.array(dataset[0][0]).shape)
    # print(data)
    return dataset, weight


# @profile
def build_dataloader(cfg):

    if torch.distributed.is_initialized():
        ddp = True
    else:
        print("ddp False")
        ddp = False

    # test_flag = False
    # if 'test_data' in cfg.keys():
    #     test_flag = True
    # else:
    #     test_flag = False

    dataloader_cfg = copy.deepcopy(cfg)
    try:
        dataloader_cfg = cfg['dataloader']
    except Exception:
        raise 'should contain {dataloader}!'

    if ddp:
        # if test_flag == False:
        train_ds, train_weight = build_dataset(cfg, 'train_data')
        # print(np.array(train_ds[0][0]).shape) 
        # print(len(train_ds))
        # print(np.array(train_ds[0][1]).shape) 
        val_ds, val_weight = build_dataset(cfg, 'val_data')
        train_sampler = DistributedSampler(train_ds)
        train_loader = DataLoader(train_ds,
                                    sampler=train_sampler,
                                    # drop_last=True,
                                    **dataloader_cfg
                                    )
        # print(train_sampler)
        # for i, data in enumerate(train_loader):
        #     with open("data.txt", "w") as f:
                
        #         np.set_printoptions(threshold=np.inf)
        #         f.write(str(data[0][0]))
        #         f.write(str(data[0][1]))
        #         f.write(str(data[0][2]))
        #         f.write(str(data[0][3]))
        #         f.write(str(data[0]))
        #     break
            # print(np.array(data[0][0]).shape)
            # print(np.array(data[0][1]).shape)
            # print(np.array(data[0][2]).shape)
            # print(np.array(data[0][3]).shape)
            
        
        val_loader = DataLoader(val_ds,
                                # drop_last=True,
                                **dataloader_cfg)
        # for i, data in enumerate(val_loader):
        #     print(np.array(data[0]).shape)
        del train_ds
        del train_sampler
        del val_ds
        del val_weight
        gc.collect()
        return train_loader, val_loader, train_weight
        # else:
        #     test_ds = build_dataset(cfg, 'test_data')
        #     test_loader = DataLoader(test_ds,
        #                              **dataloader_cfg)

        #     return None, test_loader
    else:
        # if test_flag == False:
        val_ds = build_dataset(cfg, 'val_data')

        val_loader = DataLoader(val_ds,
                                **dataloader_cfg)

        return None, val_loader
        # else:
        #     test_ds = build_dataset(cfg, 'test_data')
        #     test_loader = DataLoader(test_ds,
        #                              **dataloader_cfg)

        #     return None, test_loader
        
        
# @profile
# def build_train_dataloader(cfg):

#     if torch.distributed.is_initialized():
#         ddp = True
#     else:
#         print("ddp False")
#         ddp = False

#     # test_flag = False
#     # if 'test_data' in cfg.keys():
#     #     test_flag = True
#     # else:
#     #     test_flag = False

#     dataloader_cfg = copy.deepcopy(cfg)
#     try:
#         dataloader_cfg = cfg['dataloader']
#     except Exception:
#         raise 'should contain {dataloader}!'

#     if ddp:
#         train_ds, train_weight = build_dataset(cfg, 'train_data')
#         # print(np.array(train_ds[0][0]).shape)
#         train_sampler = DistributedSampler(train_ds)
#         train_loader = DataLoader(train_ds,
#                                     sampler=train_sampler,
#                                     **dataloader_cfg
#                                     )
#         # X, y = next(iter(train_loader))
#         # print(np.array(X).shape)
#         del train_ds
#         del train_sampler
#         gc.collect()
#         return train_loader, train_weight
        
# def build_val_dataloader(cfg):

#     dataloader_cfg = copy.deepcopy(cfg)
#     try:
#         dataloader_cfg = cfg['dataloader']
#     except Exception:
#         raise 'should contain {dataloader}!'

        
#     val_ds, val_weight = build_dataset(cfg, 'val_data')

#     val_loader = DataLoader(val_ds,
#                             **dataloader_cfg)

    
#     del val_ds
#     del val_weight
#     gc.collect()
#     return val_loader
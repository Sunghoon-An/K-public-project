import pandas as pd, pickle
import time
import numpy as np
from columns_list import *
from utils import *
from config import *
import argparse
import os
from multiprocessing import Pool
from itertools import product
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

import pickle
import datetime
import math

def data_load(DATA_DIR):
    def custom_split(unique_c, cnt):
        card_list = []
        np.random.shuffle(unique_c)
        sizes = math.ceil(unique_c.size / cnt)
        for i in range(0, unique_c.size, sizes):
            card_list.append(list(unique_c[i : i + sizes]))
            print(unique_c[i : i + sizes].size)
        return card_list
            
    df = dd.read_csv(os.path.join(DATA_PATH, 'df.csv')
                     , dtype = dtypes
#                      , encoding = 'UTF-8'
                     , engine = 'python')
    df[UNIQ_COLS] = df[UNIQ_COLS].astype(np.object)
    df[CODE_COLS] = df[CODE_COLS].astype(np.object)
    df[AMT_TYPE] = df[AMT_TYPE].astype(np.float)
    
    unique_usr = df.RCMS_SBJT_ID.unique().compute(scheduler = 'multiprocessing', num_worker = 4)
    usr_list = custom_split(unique_usr, 48)
    
    for i, usrs in enumerate(usr_list):
        data = df[df.AGRT_ORGN_ID.isin(usrs)].reset_index(drop = True)
        print('========== Reset index Done ==========')
        data = data.compute(scheduler = 'multiprocessing', num_worker = 4)
        print('========== Transform Pandas Done ==========')
        save_file = os.path.join(DATA_PATH, 'raw_dataset_{}.pkl'.format(str(i).zfill(2)))
        with open(save_file, 'wb')as f:
            pickle.dump(data, f)

if __name__=="__main__":
    data_load(DATA_PATH)
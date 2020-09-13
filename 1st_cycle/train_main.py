import pandas as pd
import numpy as np
import glob
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random
from random import randint
import os, sys, json, pickle, time, datetime, logging
from collections import Counter
import argparse, re
from threading import Thread, RLock
lock = RLock()
import multiprocessing as mp
from multiprocessing import Pool, Pipe, Process, Lock, Manager, current_process

from config import *
from utils import *
from columns_list import *
from models import *
warnings.filterwarnings("ignore")
from tqdm import tqdm
tqdm.pandas()

def train(logger):
    start = datetime.datetime.now().replace(microsecond = 0)
    logger.info("============== Modeling start ==============")
    logger.info("[start] : {}".format(str(start)))
    logger.info("DATA_PATH : {}".format(DATA_PATH))

    ##########################################
    # Config
    ##########################################
    savepath = args.savepath
    fold = args.fold_num
    random = args.random
    data_path = args.data_path
    val_path = args.val_path
    model = args.model
    sample_type = args.sample_type
    over_method = args.over_method
    under_method = args.under_method
    nb_class = args.nb_class
    opts = args.opts
    lossfunc = args.lossfunc
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    callback_type = args.callback_type
    restore = args.restore
    epoch = args.epoch
    scaler = args.scaler
    jobs = args.jobs
    ##########################################
    # Start
    ##########################################
    
    with open(os.path.join(RESULT_PATH, 'col.txt'), 'rb')as f:
        use_col = pickle.load(f)
    column_order = list(use_col)
    column_order.append('target')

    # Make dir
    if not random:
        save_path = os.path.join(savepath, "fold_{}".format(fold))
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        path = os.path.join(save_path, model)
        if os.path.isdir(path) == False:
            os.mkdir(path)
        tensorpath = os.path.join(path, "tensorboard")
        if os.path.isdir(tensorpath) == False:
            os.mkdir(tensorpath)

        fold_dir = os.path.join(FOLD_BASE_PATH, "fold_{}".format(fold))

        # load Data
        logger.info("pwd is {}".format(fold_dir))
        train_file = os.path.join(fold_dir, data_path)
        val_file = os.path.join(fold_dir, val_path)
    else:
        path = os.path.join(savepath, model)
        
        if os.path.isdir(path) == False:
            os.mkdir(path)
        tensorpath = os.path.join(path, "tensorboard")
        if os.path.isdir(tensorpath) == False:
            os.mkdir(tensorpath)
        logger.info("pwd is {} & {}".format(FOLD_BASE_PATH, data_path))
        train_file = os.path.join(FOLD_BASE_PATH, data_path)
        val_file = os.path.join(FOLD_BASE_PATH, val_path)
    
    train = pd.read_csv(train_file, low_memory = False)
    test = pd.read_csv(val_file, low_memory = False)

    logger.info("[train] - Initial Shape is : {}".format(train.shape))
    logger.info("[test] - Initial Shape is : {}".format(test.shape))
    
#     train = train.sample(frac = 0.1)
    test = test.sample(frac = 0.3)

    
    x_train = train.drop('target', axis = 1)
    y_train = train['target']
    
    x_test = test.drop('target', axis = 1)
    y_test = test['target']
    x_reffer = x_train.copy()
    
    logger.info("[Train y value is] : {}".format(y_train.value_counts()))
    logger.info("[Test y value is] : {}".format(y_test.value_counts()))
    
    # Scaling
    logger.info("============== Scaling start ==============")
    scaler = scale_choose(scaler)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    scaler_file = os.path.join(savepath, 'scaler_{}.pkl'.format(model))
    with open(scaler_file, 'wb')as f:
        pickle.dump(scaler, f)
    
    x_train, y_train = imbalance(x_train = x_train
                                 , y_train = y_train
                                 , types = sample_type
                                 , over = over_method
                                 , under = under_method
                                 , logger = logger
                                 , jobs = jobs
                                 , reffer = x_reffer
                                )
    
    del x_reffer
    

    end = datetime.datetime.now().replace(microsecond = 0)
    logger.info("[model preparing time] : {}".format(end - start))
    
    # Training
    logger.info("============== Training start ==============")
    start = datetime.datetime.now().replace(microsecond = 0)
    logger.info("[start] : {}".format(str(start)))
    
    
    
    ModelTrain(data = (x_train, y_train)
               , nb_class = nb_class
               , opts = opts
               , lossfunc = lossfunc
               , model = model
               , batch_size = batch_size
               , val_data = (x_test, y_test)
               , learning_rate = learning_rate
               , savepath = savepath
               , callback_type = callback_type
               , restore = restore
               , epoch = epoch
               , logger = logger
              )
    
    end = datetime.datetime.now().replace(microsecond = 0)
    logger.info("[model preparing time] : {}".format(end - start))
               
if __name__ == '__main__':
    logger = get_logger(os.path.basename('train'))
    args = args()
    
    train(logger)
    
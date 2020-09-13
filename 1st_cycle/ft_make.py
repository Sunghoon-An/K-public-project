import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np

import os, sys, json, pickle, time, datetime, argparse, logging

from config import *
from feature import *
from utils import *
import featuretools as ft
import featuretools.variable_types as vtypes

warnings.filterwarnings("ignore")
csv_file = os.path.join(DATA_PATH, '$$$.csv')

with open(os.path.join(DATA_PATH, 'use_col.txt'), 'rb')as f:
    use_col = pickle.load(f)
column_order = list(use_col)
df = pd.read_csv(csv_file, usecols = column_order, names = column_order, index_col = 0)
logger = get_logger(os.path.basename('Auto feature'))
start = time.time()
logger.info("==============Feature creation start==============")
logger.info("[start] : {}".format(str(start)))
logger.info("DATA_PATH : {}".format(DATA_PATH))

df = df.reset_index(drop = True)

orgn_bool = ["$$$",
"$$$",
"$$$",
"$$$"]
o_bool = []
for c in orgn_bool:
    k = [col for col in df.columns if c in col]
    o_bool+=k

orgn_ord = ["$$$",
"$$$",
"$$$",
"$$$"]
o_ord = []
for c in orgn_ord:
    k = [col for col in df.columns if c in col]
    o_ord+=k

orgn_cont = ["$$$",
"$$$",
"$$$",
"$$$",
"$$$"]
o_cont = []
for c in orgn_cont:
    k = [col for col in df.columns if c in col]
    o_cont += k

etc_bool = ["$$$",
"$$$",
"$$$",
"$$$"]
e_bool = []
for c in etc_bool:
    k = [col for col in df.columns if c in col]
    e_bool += k

etc_ord = ["$$$",
"$$$"]
e_ord = []
for c in etc_ord:
    k = [col for col in df.columns if c in col]
    e_ord += k

etc_cont = ["$$$",
"$$$",
"$$$",
"$$$"]
e_cont = []
for c in etc_cont:
    k = [col for col in df.columns if c in col]
    e_cont += k

target_bool = ["$$$",
"$$$",
"$$$",
"$$$",
"$$$"]
t_bool = []
for c in target_bool:
    k = [col for col in df.columns if c in col]
    t_bool += k

target_cont = ["$$$",
"$$$",
"$$$",
"$$$",
"$$$"]
t_cont = []
for c in target_cont:
    k = [col for col in df.columns if c in col]
    t_cont += k

orgn_col = o_bool + o_ord + o_cont 
etc_col = e_bool + e_ord + e_cont
target_col = t_bool + t_cont

logger.info("[column checked]")

df['USE_DE'] = df['USE_DE'].apply(lambda x : pd.to_datetime(x, infer_datetime_format = True))

for v in (o_bool + e_bool + t_bool):
    df[v] = df[v].astype('bool')
for v in (o_cont + e_cont + t_cont):
    df[v] = df[v].astype(float)
for v in (o_ord + e_ord):
    try:
        df[v] = df[v].astype(int)
    except Exception as e:
        print('Could not convert {v} because of missing values.')


        
from collections import Counter
intersection = Counter(orgn_col) & Counter(target_col)
multiset_a_without_common = Counter(orgn_col) - intersection
multiset_b_without_common = Counter(target_col) - intersection
orgn_col = list(multiset_a_without_common.elements())
target_col = list(multiset_b_without_common.elements())

import collections
print(len(orgn_col))
print(len(etc_col))
print(len(target_col))
orgn_col = list(dict.fromkeys(orgn_col))
etc_col = list(dict.fromkeys(etc_col))
target_col = list(dict.fromkeys(target_col))
print(len(orgn_col))
print(len(etc_col))
print(len(target_col))
logger.info("data Columns before entity is : {}".format(df.shape))
es = ft.EntitySet(id = 'data_id')
es.entity_from_dataframe(entity_id = 'data'
                         , dataframe = df
                         , make_index = True
                         , index = 'target_id'
                         , time_index = 'USE_DE')

es.normalize_entity(base_entity_id = 'data'
                   , new_entity_id = 'orgn'
                   , index = 'RCMS_BSNS_ID'
                   , make_time_index = False
                   , additional_variables = orgn_col)

es.normalize_entity(base_entity_id = 'data'
                   , new_entity_id = 'target'
                   , index = 'RECHCT_USE_ITEPD_ID'
                   , make_time_index = False
                   , additional_variables = target_col + ['target'])
from featuretools.variable_types import Numeric, PandasTypes
from featuretools.primitives import make_agg_primitive

def range_calc(numeric):
    return np.max(numeric) - np.min(numeric)
range_ = make_agg_primitive(function = range_calc
                           , input_types = [PandasTypes]
                           , return_type = PandasTypes)
def p_corr_calc(numeric1, numeric2):
    return np.corrcoef(numeric1, numeric2)[0, 1]

pcorr_ = make_agg_primitive(function = p_corr_calc,
                            input_types = [PandasTypes, PandasTypes]
                            , return_type = PandasTypes)
def s_corr_calc(numeric1, numeric2):
    return spearmanr(numeric1, numeric2)[0]

scorr_ = make_agg_primitive(function = s_corr_calc, 
                           input_types = [PandasTypes, PandasTypes]
                           , return_type = PandasTypes)

feature_matrix, feature_names = ft.dfs(entityset = es
                                      , target_entity = 'target'
                                      , agg_primitives = ['min', 'max', 'mean', 'percent_true', 'num_unique'
                                                          , 'all', 'any', 'mode'
                                                         , 'sum', 'skew', 'std'
                                                          , range_, pcorr_, scorr_
                                                         ]
                                      , trans_primitives = ['percentile', 'day', 'haversine', 'weekday', 'year', 'month']
                                      , n_jobs = -1
                                       , max_depth = 2
                                      , verbose = 1
                                      , max_features = 1000)

fc = feature_matrix.columns.tolist()
dc = df.columns.tolist()

from collections import Counter
intersection = Counter(fc) & Counter(dc)
multiset_a_without_common = Counter(fc) - intersection
fc = list(multiset_a_without_common.elements())
feature_matrix = feature_matrix[fc]
logger.info("feature matrix Columns after entity is : {}".format(feature_matrix.shape))
new_df = pd.merge(df, feature_matrix, on = 'RECHCT_USE_ITEPD_ID', how = 'left')
new_df.drop('target_id', axis = 1, inplace = True)

csv_file = os.path.join(DATA_PATH, 'make_feature.csv')
df.to_csv(csv_file)

h, m, s = count_time(start)
logger.info("Total AutoFeature Creation : [{}:{}:{}]".format(h, m, s))
logger.info("===== AutoFeature Creation Done =====")

import os
import re
import logging
import time
import pickle
import math

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from numba import jit, cuda
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import SimpleImputer,IterativeImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN, SVMSMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.test.utils import datapath, get_tmpfile
from khaiii import KhaiiiApi
from tqdm import tqdm

from feature import *
from columns_list import *
from config import *

#################################
## Preprocess
#################################

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amt_method', type = str, default = 'linear_reg'
                        , choices = ['mean', 'median', 'linear_reg', 'random_forest']
                        , help = 'imputation method for amount')
    parser.add_argument('--normal_method', type = str, default = 'most'
                        , choices = ['most_frequent', 'random_forest', 'knn']
                        , help = 'imputation method for code value')
    parser.add_argument('--encoding', type = str, default = 'one_hot'
                        , choices = ['one_hot', 'frequency']
                        , help = 'Encoding what you want')
    parser.add_argument('--nlp_method', type = str, default = 'lda'
                        , choices = ['lda', 'doc2vec']
                        , help = 'Embedding Method')
    parser.add_argument('--vec_size', type = int, default = 4
                        ,  help = 'Only int to be availabe')
    
    args = parser.parse_args()
    return args

def drop_noise_row(df, logger):
    
    logger.info("===== Drop Noise Start =====")
    start = time.time()
    logger.info("[Before Drop Noise rows shape] : {}".format(df.shape[0]))
    
    data_drop_index = df[(df['$$$'] != "$$$") 
                         & (df['$$$'] != "$$$")
                         & (df['$$$'] != "$$$")].index
    df.drop(data_drop_index, axis = 0, inplace=True)
    data_drop_index = df[(df["$$$"] == "$$$") 
                           & (df["$$$"].notnull())].index
    df.drop(data_drop_index, axis = 0, inplace=True)
    
    logger.info("[After Drop Noise rows shape] : {}".format(df.shape[0]))
    h, m, s = count_time(start)
    logger.info("Total time for Drop Noise : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Drop Noise Done =====")
    
    return df

def target_labeling(df, logger):
    
    logger.info("===== Multi-Labeling Start =====")
    start = time.time()
    
    df.loc[df.$$$=="$$$","target"]=1
    df.loc[df.$$$=="$$$","target"]=2
    df.loc[df.$$$=="$$$","target"]=3

    df.target.fillna(0, inplace = True)
    df.target = df.target.astype("int")
    
    df = df.drop(["$$$","$$$"],axis = 1)
    
    h, m, s = count_time(start)
    logger.info("Total time for Multi-Labeling : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Multi-Labeling Done =====")
    
    return df

def drop_columns(df, logger, null_rate = 0.7):
    logger.info("===== Drop Column 70% ratio Start =====")
    logger.info("[Before Drop 70% Columns] : {}".format(df.shape[1]))
    
    df = df.dropna(axis = 1, thresh = df.shape[0] * (1 - null_rate))
    
    logger.info("[After Drop 70% Columns] : {}".format(df.shape[1]))
    logger.info("===== Drop Column 70% ratio Done =====")
    return df


def get_dummie(df, code_cols, value_list):
    ###### One Hot encoding ######
    for v in value_list:
        df["{}_{}".format(str(code_cols), str(v))] = (df[code_cols] == v)
    df.drop(code_cols, axis=1, inplace = True)
    return df

def frequency_encoding(df, code_cols, logger):
    ###### Frequency Encoding ######
    logger.info("===== Frequency Encoding start =====")
    start = time.time()
    total_df = None
    logger.info("[Before Frequency Encoding columns shape] : {}".format(len(code_cols)))
    
    for col in code_cols:
        fe = df.groupby(col).size() / len(df)
        tmp = df[col].map(fe)
        total_df = pd.concat([total_df,tmp],axis=1)
    total_df.columns = code_cols
    
    logger.info("[After Frequency Encoding columns shape] : {}".format(total_df.shape[1]))
    h, m, s = count_time(start)
    logger.info("Total time for Frequency Encoding : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Frequency Encoding done =====")
    return total_df


def unique_data_preprocessing(df, logger):
    ###### Unique value split from Bisuness and non-Business ######
    logger.info("===== Unique data preprocessing start =====")
    start = time.time()
    
    def make_code(x):
        if x == "82" or x == "83":
            return 1
        else:
            return 0
    df["$$$"] = df["$$$"].astype(str).str[3:5]
    df["$$$"] = df["$$$"].apply(make_code)
    
    h, m, s = count_time(start)
    logger.info("Total time for Unique data : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Unique data preprocessing Done =====")
    return df


def amt_data_preprocessing(df, amt_col):
    ###### log transformation ######
    for col in amt_col:
        log_col = str(col) + '_log'
        df[log_col] = df[col].apply(lambda x : np.log10(abs(x) + 1))
        
    return df

def add_sign(df, float_col):
    ###### Define Positive and Negative ######
    for col in float_col:
        df["{}_sign".format(col)] = df[col].apply(lambda x : 1 if x < 0 else 0)
        
    return df


def null_data_regression(df, method = None):
    
    ###### fill missing value using Imputer  ######    
    if method == "mean":
        df = df.fillna(np.nan)
        imputer = SimpleImputer(missing_values = np.nan, strategy="mean")
        return_values = imputer.fit_transform(df)
        
    elif method =="most":
        df = df.fillna(np.nan)
        imputer = SimpleImputer(missing_values = np.nan, strategy="most_frequent")
        return_values = imputer.fit_transform(df)
        
    elif method =="median":
        df = df.fillna(np.nan)
        imputer = SimpleImputer(missing_values = np.nan, strategy="median")
        return_values = imputer.fit_transform(df)

    elif method =="linear_reg":
        df = df.fillna(np.nan)
        imputer = IterativeImputer(LinearRegression(), max_iter = 10, random_state = 42)
        return_values = imputer.fit_transform(df)
        return_values = return_values[:,-1]

    elif method =="random_forest":
        df = df.fillna(np.nan)
        imputer = IterativeImputer(RandomForestRegressor(),max_iter = 10, random_state = 42)
        return_values = imputer.fit_transform(df)
        return_values = return_values[:,-1]

    elif method =="random_forest_code":
        df = df.fillna(np.nan)
        imputer = IterativeImputer(RandomForestClassifier(n_estimators = 20),max_iter = 10, random_state = 42)
        return_values = imputer.fit_transform(df)
        return_values = return_values[:,-1]
    
    elif method == "knn":
        df = df.fillna(np.nan)
        imputer = KNNImputer(n_neighbors = 3, weights = "uniform")
        return_values = imputer.fit_transform(df)
        return_values = return_values[:,-1]
    else:
        raise ValueError("Method does not exist")
        
    return return_values

def non_amt_replace(df, logger, method):
                    
    ###### Fill missing value  ######
    logger.info("===== Data Imputation start =====")
    start = time.time()
    
    col_list = []
    col_list.extend(CODE_COLS)
    col_list.extend(UNIQ_COLS)
    col_list.extend(DATETIME)
                    
    logger.info("[Before imputation number of missing values]")
    for col in col_list + NATURAL_LANGUAGE:
        num_null = df[col].isnull().sum()
        logger.info("[Column '{}'] : [{}]".format(col, num_null))
        
    logger.info("[Code, Unique, Datetime data imputation] Method : {}".format(method))
                    
    imputer = SimpleImputer(missing_values = np.nan, strategy = method)
    cols = df.columns
    df = imputer.fit_transform(df)
    df = pd.DataFrame(df, columns = cols)
    
    with open("result/non_amt_imputer.model", "wb") as f:
        pickle.dump(imputer, f)
                    
    logger.info("[Natural Language fill]".format(len(col_list)))
    df[NATURAL_LANGUAGE] = df[NATURAL_LANGUAGE].fillna("Not exist")
    
    h, m, s = count_time(start)
    logger.info("Time for data imputation : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Data Imputation Done =====")
                    
    return df
                    
def amt_replace(df, logger, method):
    ###### Fill missing value  ######
    logger.info("===== Continuos Data Imputation start =====")
    start = time.time()
    
    col_list = []
    col_list.extend(FLOAT_TYPE)
    col_list.extend(INT_TYPE)
    col_list.extend(AMT_TYPE)
    for col in col_list:
        num_null = df[col].isnull().sum()
        logger.info("[Column '{}'] : [{}]".format(col, num_null))
    
    imputer = SimpleImputer(missing_values = np.nan, strategy = method)
    cols = df.columns
    df = imputer.fit_transform(df)
    df = pd.DataFrame(df, columns = cols)
                    
    with open("result/amt_imputer.model", "wb") as f:
        pickle.dump(imputer, f)
                    
    h, m, s = count_time(start)
    logger.info("Time for Continuos imputation : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Continuos Data Imputation Done =====")
    return df

#################################
## NLP
#################################


def doc2tokens(df, nl_cols):
    
    """tokenize documents
    Args:
        df : dataframe data
        nl_cols : NL column list
    """
    
    total_val = []
    tokenizer = KhaiiiApi()
    tokenized_corpus = []
    
    def make_tokens(df):
        try:
            tokened = tokenizer.analyze(df)
        except:
            tokened = tokenizer.analyze("Not exist")
        for word in tokened:
            tokens = [str(m).split("/")[0] for m in word.morphs]
        tokenized_corpus.append(tokens)
        
    for col in nl_cols:
        df[col].apply(make_tokens)
    
    return tokenized_corpus

def lda_make_array(model, data, vec_size):
    
    """ make lda model's values 2-dim numpy array
    Args :
        model : LDA model object
        vec_size : vector size
    """
    
    num_list = []
    
    for i in tqdm(range(len(data))):
        tmp_val = model.get_document_topics(data[i])
        tmp_list = [0]*vec_size
        vals_sum = 0
        for j in range(len(tmp_val)):
            idx = tmp_val[j][0]
            val = tmp_val[j][1]
            vals_sum += val
            tmp_list[idx] = val
        null_num = vec_size - len(tmp_val)
        if null_num > 0:
            for j in range(len(tmp_list)):
                if tmp_list[j] == 0:
                    tmp_list[j] = (1-vals_sum) / null_num
        num_list.append(tmp_list)
    total_val = np.array(num_list)
    return total_val

def lda_model(tokenized_corpus, vec_size):
    
    """create LDA model 
    Args:
        tokenized_corpus : tokenized documents
        vec_size : vector size
    """
    
    dictionary = corpora.HashDictionary(tokenized_corpus)
    corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]
    temp_file = os.path.join(NLP_PATH,"lda_model")
    if os.path.isfile(temp_file):
        LDA = LdaModel.load(temp_file)#LdaMulticore.load(temp_file)
    else:
        LDA = LdaModel(corpus,id2word=dictionary,num_topics=vec_size)
        LDA.save(temp_file)
    return LDA,corpus


def doc2vec_model(tokenized_corpus, vec_size):
    """create doc2vec model
    Args:
        tokenized_corpus : tokenized documents
        vec_size : vector size
    """
    total_documents = []
    cols_num = 0
    total_documents = [TaggedDocument(doc,[i]) for i, doc in enumerate(tokenized_corpus)]
    temp_file = os.path.join(NLP_PATH,"doc2vec_model")
    if os.path.isfile(temp_file):
        model = Doc2Vec.load(temp_file)
    else:
        model = Doc2Vec(total_documents,negative=5,vector_size=vec_size,window=3,workers=64)
    #model.train(total_documents,total_examples=len(total_documents),epochs=50,start_alpha=0.0001,end_alpha=0.01)
    model.save(temp_file)
    return model

def doc2vec_make_array(model, data_length):
    """ make doc2vec model's values 2-dim numpy array
    Args :
        model : doc2vec model
        data_length : the length of dataframe's rows
    """
    num_list = []
    for i in tqdm(range(data_length)):
        num_list.append(model.docvecs[i])
    total_val = np.array(num_list)
    return total_val

def array2dataframe(arr, nl_cols, vec_size, data_length):
    """ make 2-dim numpy array dataframe
    Args :
        arr : 2-dim numpy array
        nl_cols : NL columns list
        vec_size : vector size
        data_length : the length of dataframe's rows
    """
    arr_val = []
    for i in range(arr.shape[0] // data_length):
        arr_val.append(arr[data_length * i:data_length * (i + 1)])
    arr_val = np.array(arr_val)
    arr_val = arr_val.transpose(1,0,2)
    arr_val = arr_val.reshape(data_length,-1)
    nl_col_list = []
    for col in nl_cols:
        for i in range(vec_size):
            nl_col_list.append(col+"_"+str(i))
    return pd.DataFrame(arr_val,columns=nl_col_list)

def nlp_processing(df, nl_cols, method, vec_size, logger):
    
    """ total NL preprocessing
    Args :
        df : DataFrame data
        nl_cols : NL columns list
        method : NL embedding method either 'lda' or 'doc2vec'
        vec_size : vector size
    """
    
    logger.info("===== Natural Language Processing Start =====")
    start = time.time()
    logger.info("[Before Natural Language shape] : {}".format(df.shape))
    logger.info("[Tokenizing Documents]")
    tokenized_corpus = doc2tokens(df, nl_cols)
    
    if method == "lda":
        model,corpus = lda_model(tokenized_corpus, vec_size)
        logger.info("[LDA model build]")
        nlp_arr = lda_make_array(model, corpus, vec_size)
    elif method == "doc2vec":
        model = doc2vec_model(tokenized_corpus,vec_size)
        logger.info("[Doc2Vec model build]")
        nlp_arr = doc2vec_make_array(model,len(tokenized_corpus))  
        
    logger.info("[Array to dataframe]")
    
    total_df = array2dataframe(nlp_arr, nl_cols, vec_size, df.shape[0])
    logger.info("[check Shape - total] : {}".format(total_df.shape))
    logger.info("[check Shape - df] : {}".format(df.shape))
    df = df.drop(nl_cols,axis=1)
    df = pd.concat([df,total_df],axis=1)
    
    logger.info("[After Natural Language shape] : {}".format(df.shape))
    h, m, s = count_time(start)
    logger.info("Total time for NLP : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Natural Language Processing Done =====")
    
    return df















# coding: utf-8

# In[1]:

import numpy as np
import collections
import pandas as pd
import pyodbc
import time
import pickle
import operator
from operator import itemgetter
from joblib import Parallel, delayed

from sklearn import linear_model
import statsmodels.formula.api as sm
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine

import psycopg2
from sklearn.utils import shuffle

import sql
from sklearn import feature_selection

from sklearn import linear_model
import statsmodels.formula.api as sm
from statsmodels.stats import anova
import pylab as pl

import warnings
from sqlalchemy.pool import NullPool
from multiprocessing import Pool
from functools import partial
from pysal.spreg.twosls import TSLS
from decimal import *
from statsmodels import robust

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import random

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf
import csv

from rpy2.robjects import pandas2ri
pandas2ri.activate()

import rpy2.robjects as robjects
from rpy2.robjects import DataFrame, Formula

from sklearn.metrics import mean_squared_error

def construct_sec_order(arr): 
    second_order_feature = []
    num_cov_sec = len(arr[0])
    for a in arr:
        tmp = []
        for i in range(num_cov_sec):
            for j in range(i+1, num_cov_sec):
                tmp.append( a[i] * a[j] )
        second_order_feature.append(tmp)
        
    return np.array(second_order_feature).sum(axis = 1).reshape((2000,1))

def data_generation_dense_2(covariance, num_control, num_treated, num_cov, pi, control_m = 0.1, treated_m = 0.9):

    def gen_xz(num_treated, num_control, dim):
        z = np.concatenate((np.zeros(num_control), np.ones(num_treated)), axis = 0).reshape((num_treated + num_control, 1))

        x1_0 = np.random.binomial(1,0.5,size = (num_control,dim-0))
        x1_1 = np.random.binomial(1,0.1,size = (num_control,0))
        x1 = np.hstack((x1_0,x1_1))

        x2_0 = np.random.binomial(1,0.5,size = (num_treated,dim-0))
        x2_1 = np.random.binomial(1,0.9,size = (num_treated,0))
        x2 = np.hstack((x2_0,x2_1))

        x = np.concatenate((x1, x2), axis = 0)

        return x, z
    
    def get_gamma(num_cov, ratio, base):
        gamma = []
        for i in range(num_cov):
            gamma.append(base)
            base = base * ratio
        return gamma
            
    #parameters
    alpha = 0
    k = 0  
    #beta_hat = 5.5
    dim = num_cov
    mean_rou = 0.1
    rou = np.random.normal(mean_rou, mean_rou / 10, dim).reshape((dim,1))
    epsilon_ksi = np.random.multivariate_normal([0,0], [[1, covariance], [covariance, 1]], num_control + num_treated)
    
    x,z = gen_xz(num_treated, num_control, dim)
    xz = np.concatenate((x, z), axis =1)

    dij = np.add(pi * z, np.matmul(x, rou))
    dij = np.add(dij, epsilon_ksi[:,1].reshape((num_treated + num_control,1)))
    
    threshold1 = 0.3
    threshold2 = 0.6
    threshold3 = 1.0
    Dij = np.asarray([0 if e < threshold1 else 1 if e < threshold2 else 2 if e < threshold3 else 3 for e in dij[:,0]]).reshape((num_treated + num_control,1))

    s = np.random.uniform(-1,1)
    U = 1
    alpha = np.random.normal(10 * s, 1, 10)
    beta = np.random.normal(1.5,0.15,10)
    gamma = get_gamma(10,0.5,5)
    #Rij = np.add(beta_hat * Dij, np.matmul(x, gamma).reshape((num_treated + num_control,1)))
    Rij = np.add(np.matmul(x, alpha).reshape((num_treated + num_control,1)), Dij * np.matmul(x,beta).reshape((num_treated + num_control,1)))
    Rij = np.add(Rij, U * construct_sec_order(x[:,:5]))
    Rij = np.add(Rij, epsilon_ksi[:,0].reshape([num_treated + num_control,1]))  

    df = pd.DataFrame(np.concatenate([x, z, Dij, Rij], axis = 1)) 
    df.columns = df.columns.astype(str)
    df.rename(columns = {'10':"iv"}, inplace = True)
    df['iv'] = df['iv'].astype('int64')  
    df.rename(columns = {'11':"treated"}, inplace = True)
    df['treated'] = df['treated'].astype('int64') 
    df.rename(columns = {'12':"outcome"}, inplace = True)

    df['zr'] = df['iv'] * df['outcome']
    df['zd'] = df['iv'] * df['treated']
    df['matched'] = 0

    treatment_effect = np.matmul(x,beta).reshape((num_treated+num_control,1))
    df['treatment_effect'] = treatment_effect

    df = df.reset_index()

    return df,x,z,Dij,Rij

# this function takes the current covariate list, the covariate we consider dropping, name of the data table, 
# name of the holdout table, the threshold (below which we consider as no match), and balancing regularization
# as input; and outputs the matching quality
def score_tentative_drop_c(cov_l, c, db_name, holdout_df, thres = 0, tradeoff = 1.0):
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='yaoyj11 '")
    cur = conn.cursor() 
    
    covs_to_match_on = set(cov_l) - {c} # the covariates to match on
    
    # the flowing query fetches the matched results (the variates, the outcome, the treatment indicator)
    s = time.time()
    
    cur.execute('''with temp AS 
        (SELECT 
        {0}
        FROM {3}
        where "matched"=0
        group by {0}
        Having sum("iv") > 0 and sum("iv") < count(*)
        )
        (SELECT {1}, iv,treated, outcome
        FROM {3}
        WHERE "matched"=0 AND EXISTS 
        (SELECT 1
        FROM temp 
        WHERE {2}
        )
        )
        '''.format(','.join(['"{0}"'.format(v) for v in covs_to_match_on ]),
                   ','.join(['{1}."{0}"'.format(v, db_name) for v in covs_to_match_on ]),
                   ' AND '.join([ '{1}."{0}"=temp."{0}"'.format(v, db_name) for v in covs_to_match_on ]),
                   db_name
                  ) )
    res = np.array(cur.fetchall())
    
    time_match = time.time() - s
    
    s = time.time()
    # the number of unmatched treated units
    cur.execute('''select count(*) from {} where "matched" = 0 and "iv" = 0'''.format(db_name))
    num_control = cur.fetchall()
    # the number of unmatched control units
    cur.execute('''select count(*) from {} where "matched" = 0 and "iv" = 1'''.format(db_name))
    num_treated = cur.fetchall()
    time_BF = time.time() - s
    
    s = time.time() # the time for fetching data into memory is not counted if use this
    
    tree_c = Ridge(alpha=0.1)
    tree_t = Ridge(alpha=0.1)
    
    holdout = holdout_df.copy()
    holdout = holdout[ ["{}".format(c) for c in covs_to_match_on] + ['iv', 'treated', 'outcome']]

    mse_t = np.mean(cross_val_score(tree_t, holdout[holdout['iv'] == 1].iloc[:,:-3], 
                                holdout[holdout['iv'] == 1]['outcome'] , scoring = 'neg_mean_squared_error' ) )
        
    mse_c = np.mean(cross_val_score(tree_c, holdout[holdout['iv'] == 0].iloc[:,:-3], 
                                holdout[holdout['iv'] == 0]['outcome'], scoring = 'neg_mean_squared_error' ) )
      
    time_PE = time.time() - s
    
    if len(res) == 0:
        return (( mse_t + mse_c ), time_match, time_PE, time_BF)
    else:        
        return (tradeoff * (float(len(res[res[:,-3]==0]))/num_control[0][0] + float(len(res[res[:,-3]==1]))/num_treated[0][0]) +             ( mse_t + mse_c ), time_match, time_PE, time_BF)
        
# update matched units
# this function takes the currcent set of covariates and the name of the database; and update the "matched"
# column of the newly mathced units to be "1"

def update_matched(cur, conn, covs_matched_on, db_name, level):  

    cur.execute('''with temp AS 
        (SELECT 
        {0}
        FROM {3}
        where "matched"=0
        group by {0}
        Having sum("iv") > 0 and sum("iv") < count(*)
        )
        update {3} set "matched"={4}
        WHERE EXISTS
        (SELECT {0}
        FROM temp
        WHERE {2} and {3}."matched" = 0
        )
        '''.format(','.join(['"{0}"'.format(v) for v in covs_matched_on]),
                   ','.join(['{1}."{0}"'.format(v, db_name) for v in covs_matched_on]),
                   ' AND '.join([ '{1}."{0}"=temp."{0}"'.format(v, db_name) for v in covs_matched_on ]),
                   db_name,
                   level
                  ) )
    conn.commit()

    return

# get CATEs 
# this function takes a list of covariates and the name of the data table as input and outputs a dataframe 
# containing the combination of covariate values and the corresponding CATE 
# and the corresponding effect (and the count and variance) as values

def get_CATE_db(cur, cov_l, db_name, level):
    cur.execute(''' select {0},count(*),sum(treated),sum(outcome),array_agg(treatment_effect) as control_effect,array_agg(index) as control_index
                    from {1}
                    where matched = {2} and iv = 0
                    group by {0}
                    '''.format(','.join(['"{0}"'.format(v) for v in cov_l]), 
                              db_name, level) )
    res_c = cur.fetchall()
       
    cur.execute(''' select {0},count(*),sum(treated),sum(outcome),array_agg(treatment_effect) as treatment_effect, array_agg(index) as treatment_index
                    from {1}
                    where matched = {2} and iv = 1
                    group by {0}
                    '''.format(','.join(['"{0}"'.format(v) for v in cov_l]), 
                              db_name, level) )
    res_t = cur.fetchall()
     
    if (len(res_c) == 0) | (len(res_t) == 0):
        return None

    cov_l = list(cov_l)

    result = pd.merge(pd.DataFrame(np.array(res_c), columns=['{}'.format(i) for i in cov_l]+['count_0','sum_treated_0','sum_outcome_0','control_effect','control_index']), 
                  pd.DataFrame(np.array(res_t), columns=['{}'.format(i) for i in cov_l]+['count_1','sum_treated_1','sum_outcome_1','treatment_effect','treatment_index']),
                  on = ['{}'.format(i) for i in cov_l], how = 'inner') 
    
   
    result_df = result[['{}'.format(i) for i in cov_l] + ['count_0','sum_treated_0','sum_outcome_0','control_effect','control_index','count_1','sum_treated_1','sum_outcome_1','treatment_effect','treatment_index']]
    
    if result_df is None or result_df.empty:
        return None
    
    result_df['count'] = result_df['count_0'] + result_df['count_1']
    result_df['sum_outcome_1'] = result_df['sum_outcome_1'].astype('float64')
    result_df['sum_outcome_0'] = result_df['sum_outcome_0'].astype('float64')
    result_df['sum_treated_1'] = result_df['sum_treated_1'].astype('float64')
    result_df['sum_treated_0'] = result_df['sum_treated_0'].astype('float64')
    result_df['CITT_y'] = result_df['sum_outcome_1'] * 1.0 / result_df['count_1'] - result_df['sum_outcome_0'] * 1.0 / result_df['count_0']
    result_df['CITT_t'] = result_df['sum_treated_1'] * 1.0 / result_df['count_1'] - result_df['sum_treated_0'] * 1.0 / result_df['count_0']
    result_df = result_df.loc[result_df['CITT_t'] != 0]
    result_df['CCACE'] = result_df['CITT_y'] * 1.0 / result_df['CITT_t']
    result_df['true_effect'] = result_df['treatment_effect'] + result_df['control_effect']
    result_df['index'] = result_df['treatment_index'] + result_df['control_index']
    index = ['count','CCACE','true_effect','index', 'treatment_index']
    result_df = result_df[index]

    result_all = result_df.sum(axis = 0)
    return result_df

def run_db(cur, conn, db_name, df, holdout_df, num_covs, reg_param = 0.1):
    cur.execute('update {0} set matched = 0'.format(db_name)) # reset the matched indicator to 0
    conn.commit()

    covs_dropped = [] # covariate dropped
    ds = []
    score_list = []
    
    level = 1
    
    cur_covs = range(num_covs) 
    init_score,_,_,_ = score_tentative_drop_c(cur_covs, None, db_name, holdout_df, tradeoff = 0.1)

    update_matched(cur, conn, cur_covs, db_name, level) # match without dropping anything
    d = get_CATE_db(cur, cur_covs, db_name, level) # get CATE without dropping anything
    ds.append(d)
    
    while len(cur_covs)>1:
        level += 1
        #print(level)

        cur.execute('''select count(*) from {} where "matched"=0 and "iv"=0'''.format(db_name))
        if cur.fetchall()[0][0] == 0:
            break
        cur.execute('''select count(*) from {} where "matched"=0 and "iv"=1'''.format(db_name))
        if cur.fetchall()[0][0] == 0:
            break
        
        best_score = -np.inf
        cov_to_drop = None

        cur_covs = list(cur_covs)
        for c in cur_covs:
            score,time_match,time_PE,time_BF = score_tentative_drop_c(cur_covs, c, db_name, 
                                                                      holdout_df, tradeoff = 0.1)
            
            if score > best_score:
                best_score = score
                cov_to_drop = c
        
        """
        if (init_score < 0 and best_score < 1.05 * init_score) or (init_score >= 0 and best_score < 0.95 * init_score):
            break  
        """

        cur_covs = set(cur_covs) - {cov_to_drop} # remove the dropped covariate from the current covariate set

        update_matched(cur, conn, cur_covs, db_name, level)
        score_list.append(best_score)
        d = get_CATE_db(cur, cur_covs, db_name, level)
        ds.append(d)
        covs_dropped.append(cov_to_drop) # append the removed covariate at the end of the covariate    
      
    return get_treatment_comparison(df, ds)

def total_concentration(df):
    Y = df["treated"]
    X0 = df["0"]
    X1 = df["1"]
    X2 = df["2"]
    X3 = df["3"]
    X4 = df["4"]
    X5 = df["5"]
    X6 = df["6"]
    X7 = df["7"]
    X8 = df["8"]
    X9 = df["9"]
    IV = df["iv"]
    formula = Formula('treated ~ x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + iv')
    formula2 = Formula('treated ~ x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9')
    dataf = DataFrame({'treated': robjects.IntVector(Y), \
                        'x0': robjects.IntVector(X0),\
                        'x1': robjects.IntVector(X1),\
                        'x2': robjects.IntVector(X2),\
                        'x3': robjects.IntVector(X3),\
                        'x4': robjects.IntVector(X4),\
                        'x5': robjects.IntVector(X5),\
                        'x6': robjects.IntVector(X6),\
                        'x7': robjects.IntVector(X7),\
                        'x8': robjects.IntVector(X8),\
                        'x9': robjects.IntVector(X9),\
                        'iv': robjects.IntVector(IV)
                        })

    fit=robjects.r.lm(formula=formula, data=dataf)
    fit2=robjects.r.lm(formula=formula2, data=dataf)
    r_frame=robjects.r.anova(fit,fit2)
    py_frame = pandas2ri.ri2py_dataframe(r_frame)
    print("total concentration parameter: " + str(py_frame.iloc[1,4]))

def is_strong_iv(df,idx):
    df = df.loc[df['index'].isin(idx)]
    Y = df["treated"]
    X0 = df["0"]
    X1 = df["1"]
    X2 = df["2"]
    X3 = df["3"]
    X4 = df["4"]
    X5 = df["5"]
    X6 = df["6"]
    X7 = df["7"]
    X8 = df["8"]
    X9 = df["9"]
    IV = df["iv"]
    formula = Formula('treated ~ x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + iv')
    formula2 = Formula('treated ~ x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9')
    dataf = DataFrame({'treated': robjects.IntVector(Y), \
                        'x0': robjects.IntVector(X0),\
                        'x1': robjects.IntVector(X1),\
                        'x2': robjects.IntVector(X2),\
                        'x3': robjects.IntVector(X3),\
                        'x4': robjects.IntVector(X4),\
                        'x5': robjects.IntVector(X5),\
                        'x6': robjects.IntVector(X6),\
                        'x7': robjects.IntVector(X7),\
                        'x8': robjects.IntVector(X8),\
                        'x9': robjects.IntVector(X9),\
                        'iv': robjects.IntVector(IV)
                        })

    #print(dataf)
    fit=robjects.r.lm(formula=formula, data=dataf)
    fit2=robjects.r.lm(formula=formula2, data=dataf)
    r_frame=robjects.r.anova(fit,fit2)
    py_frame = pandas2ri.ri2py_dataframe(r_frame)
    
    return py_frame.iloc[1,4] >= 10


def get_treatment_comparison(df, res):
    total_match_cnt = 0
    true_catt = []
    estimated_catt = []

    for res_level in res:
        if res_level is None or res_level.empty:
            continue
        for idx, row in res_level.iterrows():
            if row is None:
                continue
            idx_list = row['index']
            trt_idx_list = row['treatment_index']

            if is_strong_iv(df,idx_list):
                grp_num = row['count']
                estimated_treatment_effect = row['CCACE']
                for i in trt_idx_list:
                    true_catt.append(df[df['index'] == i]['treatment_effect'])
                    estimated_catt.append(estimated_treatment_effect)
                total_match_cnt += len(trt_idx_list)  
                
    print("total match num: " + str(total_match_cnt))              
    return true_catt, estimated_catt

def plot_treatment_comparison(x,y):
    x = [elem.values[0] for elem in x]
    print("mse: " + str(mean_squared_error(x, y)))

    np.savetxt("x_nonlinear"+".txt",x,fmt="%.8f")
    np.savetxt("y_nonlinear"+".txt",y,fmt="%.8f")
    print(mean_squared_error(x,y))

def run(pi):
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='yaoyj11 '")
    cur = conn.cursor()  
    engine = create_engine('postgresql+psycopg2://postgres:yaoyj11 @localhost/postgres', poolclass=NullPool)
    table_name = 'flame_iv_linear_exp_1_' + str(int(100*pi))
    
    ATE_list = []  
    
    df_all= pickle.load(open('data/df_nonlinear_exp_'+str(pi), "rb"))

    np.random.seed(1437)

    for i in range(6,7):
        print(pi,i)
        cur.execute('drop table if exists {}'.format(table_name))
        conn.commit()
        cov = 0.8
        df = df_all[i]
        holdout_df,x,z,d,r = data_generation_dense_2(cov,1000,1000,10,pi)  
        df.to_sql(table_name, engine)
        total_concentration(df)
        true_catt, estimated_catt = run_db(cur, conn,table_name, df, holdout_df, 10)
        plot_treatment_comparison(true_catt, estimated_catt)

if __name__ == '__main__':
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    ATE_list_bias = []
    ATE_list_mad = []
    #pi_array = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    pi = 0.8
    
    run(pi)
    

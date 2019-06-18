# coding: utf-8

# In[1]:

import numpy as np
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
from astropy.stats import median_absolute_deviation
import math

def data_generation_dense_2(covariance, num_control, num_treated, num_cov_important, num_cov_unimportant, pi, control_m = 0.1, treated_m = 0.9):

    def gen_xz(num_treated, num_control, dim):
        z = np.concatenate((np.zeros(num_control), np.ones(num_treated)), axis = 0).reshape((num_treated + num_control, 1))

        x1_0 = np.random.binomial(1,0.5,size = (num_control,num_cov_important))
        x1_1 = np.random.binomial(1,0.1,size = (num_control,num_cov_unimportant))
        x1 = np.hstack((x1_0,x1_1))

        x2_0 = np.random.binomial(1,0.5,size = (num_treated,num_cov_important))
        x2_1 = np.random.binomial(1,0.9,size = (num_treated,num_cov_unimportant))
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
    beta = 10   
    dim = num_cov_important + num_cov_unimportant
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

    gamma = get_gamma(dim,0.5,5)
    Rij = np.add(beta * Dij, np.matmul(x, gamma).reshape((num_treated + num_control,1)))
    Rij = np.add(Rij, epsilon_ksi[:,0].reshape([num_treated + num_control,1]))     

    df = pd.DataFrame(np.concatenate([x, z, Dij, Rij], axis = 1))
    df.columns = df.columns.astype(str)
    start_index = dim
    df.rename(columns = {str(start_index):"iv"}, inplace = True)
    df['iv'] = df['iv'].astype('int64')  
    df.rename(columns = {str(start_index+1):"treated"}, inplace = True)
    df['treated'] = df['treated'].astype('int64') 
    df.rename(columns = {str(start_index+2):"outcome"}, inplace = True)

    df['zr'] = df['iv'] * df['outcome']
    df['zd'] = df['iv'] * df['treated']
    df['matched'] = 0

    df = df.reset_index()

    return df,x,z,Dij,Rij

# this function takes the current covariate list, the covariate we consider dropping, name of the data table, 
# name of the holdout table, the threshold (below which we consider as no match), and balancing regularization
# as input; and outputs the matching quality
def score_tentative_drop_c(cov_l, c, db_name, holdout_df, thres = 0, tradeoff = 0.1):
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
    
    tree_c = Ridge(alpha = 0.1)
    tree_t = Ridge(alpha = 0.1)
    
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
    cur.execute(''' select {0},count(*),sum(treated),sum(outcome)
                    from {1}
                    where matched = {2} and iv = 0
                    group by {0}
                    '''.format(','.join(['"{0}"'.format(v) for v in cov_l]), 
                              db_name, level) )
    res_c = cur.fetchall()
       
    cur.execute(''' select {0},count(*),sum(treated),sum(outcome)
                    from {1}
                    where matched = {2} and iv = 1
                    group by {0}
                    '''.format(','.join(['"{0}"'.format(v) for v in cov_l]), 
                              db_name, level) )
    res_t = cur.fetchall()
     
    if (len(res_c) == 0) | (len(res_t) == 0):
        return None

    cov_l = list(cov_l)

    result = pd.merge(pd.DataFrame(np.array(res_c), columns=['{}'.format(i) for i in cov_l]+['count_0','sum_treated_0','sum_outcome_0']), 
                  pd.DataFrame(np.array(res_t), columns=['{}'.format(i) for i in cov_l]+['count_1','sum_treated_1','sum_outcome_1']), 
                  on = ['{}'.format(i) for i in cov_l], how = 'inner') 
    
   
    result_df = result[['{}'.format(i) for i in cov_l] + ['count_0','sum_treated_0','sum_outcome_0','count_1','sum_treated_1','sum_outcome_1']]
    
    if result_df is None or result_df.empty:
        return None
    
    result_df['count'] = result_df['count_0'] + result_df['count_1']
    result_df['sum_outcome_1'] = result_df['sum_outcome_1'].astype('float64')
    result_df['sum_outcome_0'] = result_df['sum_outcome_0'].astype('float64')
    result_df['sum_treated_1'] = result_df['sum_treated_1'].astype('float64')
    result_df['sum_treated_0'] = result_df['sum_treated_0'].astype('float64')
    result_df['CACE_y'] = result_df['sum_outcome_1'] * 1.0 / result_df['count_1'] - result_df['sum_outcome_0'] * 1.0 / result_df['count_0']
    result_df['CACE_t'] = result_df['sum_treated_1'] * 1.0 / result_df['count_1'] - result_df['sum_treated_0'] * 1.0 / result_df['count_0']
    
    index = ['count','CACE_y','CACE_t']

    result_df = result_df[index]

    sum_all = result_df.sum(axis = 0)

    return result_df

def run_db(cur, conn, db_name, holdout_df, num_covs, reg_param = 0.1):
    cur.execute('update {0} set matched = 0'.format(db_name)) # reset the matched indicator to 0
    conn.commit()

    init_time = time.time()

    covs_dropped = [] # covariate dropped
    ds = []
    score_list = []
    
    level = 1
    #print(level)

    cur_covs = range(num_covs) 
    
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
                
        cur_covs = set(cur_covs) - {cov_to_drop} # remove the dropped covariate from the current covariate set
        #print(cov_to_drop)

        update_matched(cur, conn, cur_covs, db_name, level)
        score_list.append(best_score)
        d = get_CATE_db(cur, cur_covs, db_name, level)
        ds.append(d)
        covs_dropped.append(cov_to_drop) # append the removed covariate at the end of the covariate    
     
    end_time = time.time()
    print(end_time - init_time)
    return ds

def get_LATE(res):
    match_list = res
    index_list = ['count','CACE_y','CACE_t']

    df_all = pd.DataFrame(columns = index_list)

    for row in match_list:
        if row is None or row.empty:
            continue
        df = pd.DataFrame(row)
        df_all = pd.concat([df_all,df],axis = 0)
    
    ATE = None
    if not df_all.empty:
        df_all['weighted_CACE_y'] = df_all['CACE_y'] * df_all['count']
        df_all['weighted_CACE_t'] = df_all['CACE_t'] * df_all['count']
        sum_all = df_all.sum(axis = 0)
        ATE = sum_all['weighted_CACE_y']/sum_all['weighted_CACE_t']

    return ATE

def get_LATE_and_CI(cur,db_name,df_original,res):
    match_list = res
    index_list = ['count','CACE_y','CACE_t']

    df_all = pd.DataFrame(columns = index_list)

    for row in match_list:
        if row is None or row.empty:
            continue
        df = pd.DataFrame(row)
        df_all = pd.concat([df_all,df],axis = 0)
    
    sum_all = df_all.sum(axis = 0)
    N = sum_all['count']

    #calculate LATE
    LATE = None
    if not df_all.empty:
        df_all['weighted_CACE_y'] = df_all['CACE_y'] * df_all['count']
        df_all['weighted_CACE_t'] = df_all['CACE_t'] * df_all['count']
        sum_all = df_all.sum(axis = 0)
        LATE = sum_all['weighted_CACE_y']/sum_all['weighted_CACE_t']

    if LATE is None:
        return LATE, None, None

    #calculate 95% CI
    cur.execute(''' select array_agg(index), avg(outcome), avg(treated), avg(iv)
                from {0}
                where matched != 0
                '''.format(db_name))
    res_matched = cur.fetchall()
    df_matched = df_original[df_original['index'].isin(res_matched[0][0])][["outcome","treated","iv"]]
    df_matched['outcome'] = df_matched['outcome'].astype('float64')
    df_matched['treated'] = df_matched['treated'].astype('float64')
    df_matched['iv'] = df_matched['iv'].astype('float64')

    df_original['epsilon'] = df_original['outcome'] - df_original['outcome'].mean() - LATE * (df_original['treated'] - df_original['treated'].mean())
    df_original['var_iv'] = df_original['iv'] - df_original['iv'].mean()
    df_original['epsilon_square'] = df_original['epsilon'] * df_original['epsilon']
    df_original['var_iv_square'] = df_original['var_iv'] * df_original['var_iv']
    df_original['var_epsilon_square'] = df_original['var_iv_square'] * df_original['epsilon_square']
    matched_all = df_original.sum(axis = 0)
    cov_treated_iv = df_original.cov().loc["treated", "iv"]
    std = math.sqrt(matched_all['var_epsilon_square'] * 1.0 / ( N * cov_treated_iv ))

    return LATE, matched_all['var_epsilon_square'], N, cov_treated_iv * cov_treated_iv, std

def run(pi):
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='yaoyj11 '")
    cur = conn.cursor()  
    engine = create_engine('postgresql+psycopg2://postgres:yaoyj11 @localhost/postgres', poolclass=NullPool)
    table_name = 'flame_' + str(int(100*pi))
    
    LATE_list = []  
    
    df_all= pickle.load(open('data/df_linear_6000_multilevel_'+str(pi), "rb"))

    np.random.seed(10)
    for i in range(1):
        #print(pi,i)
        cur.execute('drop table if exists {}'.format(table_name))
        conn.commit()
        cov = 0.8
        df = df_all[i]
        num_cov = 30
        holdout_df,x,z,d,r = data_generation_dense_2(cov,3000,3000,15,5,pi)  
        df.to_sql(table_name, engine)
        res = run_db(cur, conn,table_name, holdout_df, 20)
        LATE = get_LATE(res)
        LATE_list.append(LATE)

    return LATE_list

if __name__ == '__main__':
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    mean_bias = []
    median_bias = []
    mean_deviation = []
    median_deviation = []

    #pi_array = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    pi_array = [0.5]
    
    with Pool(min(10, len(pi_array))) as p:
        drop_results = p.map(run, pi_array)
        for late_list in drop_results:
            late_list_bias = [abs(elem - 10) for elem in late_list]
            median_bias.append(np.median(late_list_bias))
            median_deviation.append(median_absolute_deviation(late_list))
    
    median_bias = [x for x in median_bias if x != 'nan']
    median_deviation = [x for x in median_deviation if x != 'nan']    
    #print(median_bias)
    #print(median_deviation)
    #pickle.dump(ATE_list, open('result/result_full_matching', 'wb'))
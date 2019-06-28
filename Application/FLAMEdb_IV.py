# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import os
import pyodbc
import time
import pickle
import operator
from operator import itemgetter
#from joblib import Parallel, delayed

from sklearn import linear_model
#import statsmodels.formula.api as sm
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine

import psycopg2
from sklearn.utils import shuffle

#import sql
from sklearn import feature_selection

from sklearn import linear_model
#import statsmodels.formula.api as sm
#from statsmodels.stats import anova
import pylab as pl

import warnings
import math
from sqlalchemy.pool import NullPool
from multiprocessing import Pool
from functools import partial
#from pysal.spreg.twosls import TSLS
from decimal import *
#from statsmodels import robust
#from astropy.stats import median_absolute_deviation
#from variance_tests_2 import * # standard error 
from late_estimators import * # standard error 

def get_data(name,outcome):
    df = pickle.load(open('data/'+ name , "rb"))

    df['iv'] = df['iv'].astype('int64')  
    df['treated'] = df['treated'].astype('int64') 

    df['zr'] = df['iv'] * df['outcome']
    df['zd'] = df['iv'] * df['treated']
    df['matched'] = 0
    df['group_id'] = 0
    
    print("total units:", df.shape[0])
    return df



def project_hold_data(holdout_df,covs_to_match_on) :
    
    #filter_col = [col for col in holdout_df if col.startswith('stratum')] # list comprehension
    
    #covs_to_match_on = covs_to_match_on - set(["stratum_identifier"])
    to_match = list(covs_to_match_on)
    to_match = map(str, to_match)
    cov_include = to_match + ["index","iv","treated","outcome"]
    
    holdout_df = holdout_df[cov_include]
    holdout_df = pd.get_dummies(holdout_df) # dummies of categorical covs
    
    return holdout_df

# this function takes the current covariate list, the covariate we consider dropping, name of the data table, 
# name of the holdout table, the threshold (below which we consider as no match), and balancing regularization
# as input; and outputs the matching quality

def score_tentative_drop_c(cov_l, c, db_name, holdout_df, thres = 0, tradeoff = 0.1):
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='yaoyj11 '")
    cur = conn.cursor() 
    
    covs_to_match_on = set(cov_l) - {c} # the covariates to match on
    #covs_to_match_on.add("stratum_identifier")

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
    
    tree_c = Ridge(alpha = 10)
    tree_t = Ridge(alpha = 10)
    
    #holdout = factorize_strata_controls(holdout_df[ ["{}".format(c) for c in covs_to_match_on] + ['iv', 'treated', 'outcome']])
    holdout = project_hold_data(holdout_df,covs_to_match_on)

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
    covs_matched_on = set(covs_matched_on)
    #covs_matched_on.add("stratum_identifier")

    cur.execute('''with temp AS 
        (SELECT {0},
        ROW_NUMBER() OVER (ORDER BY {0}) as "group_id"        
        FROM {3}
        where "matched"=0
        group by {0}
        Having sum("iv") > 0 and sum("iv") < count(*)
        )
        
        update {3} 
        set "matched"={4}, "group_id"=temp."group_id"
        FROM temp
        WHERE {3}."matched" = 0 and {2}
        '''.format(','.join(['"{0}"'.format(v) for v in covs_matched_on]), #0
                   ','.join(['{1}."{0}"'.format(v, db_name) for v in covs_matched_on]),
                   ' AND '.join([ '{1}."{0}"=temp."{0}"'.format(v, db_name) for v in covs_matched_on ]),
                   db_name, #3
                   level #4
                  ) )


#    cur.execute('''with temp AS 
#        (SELECT 
#        {0}
#        FROM {3}
#        where "matched"=0
#        group by {0}
#        Having sum("iv") > 0 and sum("iv") < count(*)
#        )
#        update {3} set "matched"={4}
#        WHERE EXISTS
#        (SELECT {0}
#        FROM temp
#        WHERE {2} and {3}."matched" = 0
#        )
#        '''.format(','.join(['"{0}"'.format(v) for v in covs_matched_on]),
#                   ','.join(['{1}."{0}"'.format(v, db_name) for v in covs_matched_on]),
#                   ' AND '.join([ '{1}."{0}"=temp."{0}"'.format(v, db_name) for v in covs_matched_on ]),
#                   db_name,
#                   level
#                  ) )
#
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

    return result_df

def run_db(cur, conn, db_name, holdout_df, num_covs, dictionary, reg_param = 0.1):
    cur.execute('update {0} set matched = 0'.format(db_name)) # reset the matched indicator to 0
    conn.commit()

    covs_dropped = [] # covariate dropped
    ds = []
    score_list = []
    
    level = 1
    #print(level)

    cur_covs = range(num_covs) 
    
    update_matched(cur, conn, cur_covs, db_name, level) # match without dropping anything
    d = get_CATE_db(cur, cur_covs, db_name, level) # get CATE without dropping anything
    ds.append(d)
    
    no_cov = len(cur_covs)
    while len(cur_covs)>1:
        print(len(cur_covs))
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

        # early stopping
            # stop if MQ drops by more than 5%
        if no_cov == len(cur_covs):     
            mq_previous = best_score
        if no_cov > len(cur_covs):         
            mq_current = best_score
            change_mq = (mq_current - mq_previous)*100/mq_previous
            print(change_mq)
            mq_previous =  mq_current
            if abs(change_mq) > 5.0 :
                return ds

                
        cur_covs = set(cur_covs) - {cov_to_drop} # remove the dropped covariate from the current covariate set
        print(  "covs to drop " + str(cov_to_drop) + " " + dictionary.get(str(cov_to_drop) )  )
        #print(best_score)


        update_matched(cur, conn, cur_covs, db_name, level)
        score_list.append(best_score)
        d = get_CATE_db(cur, cur_covs, db_name, level)
        ds.append(d)
        covs_dropped.append(cov_to_drop) # append the removed covariate at the end of the covariate    
      
    return ds

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
    print("total matched:", N)

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

    #outcome_bar = float(res_matched[0][1])
    #treated_bar = float(res_matched[0][2])
    iv_bar = float(res_matched[0][3])
    df_matched['epsilon'] = df_original['outcome'] - df_original['outcome'].mean() - LATE * (df_original['treated'] - df_original['treated'].mean())
    df_matched['var_iv'] = df_matched['iv'] - iv_bar
    df_matched['epsilon_square'] = df_matched['epsilon'] * df_matched['epsilon']
    df_matched['var_iv_square'] = df_matched['var_iv'] * df_matched['var_iv']
    df_matched['var_epsilon_square'] = df_matched['var_iv_square'] * df_matched['epsilon_square']
    matched_all = df_matched.sum(axis = 0)
    cov_treated_iv = df_matched.cov().loc["treated", "iv"]
    covariance = matched_all['var_epsilon_square'] * 1.0 / ( N * cov_treated_iv * cov_treated_iv )
    std = math.sqrt(covariance / N)

    return LATE, std


def assing_gr_id(conn) :
    
    query = '''WITH temp AS                
    (
    SELECT matched, group_id,
    ROW_NUMBER() OVER (ORDER BY matched, group_id) as "group_id_overall"
    FROM   real_exp                  
    where "matched"!=0
    group by matched, group_id
    )

    SELECT *
    FROM real_exp, temp
    WHERE real_exp.matched = temp.matched and real_exp.group_id = temp.group_id
    
    '''
    df_grp = pd.read_sql(query, conn)
    
    return df_grp
    
    

def run(sample_range,outcome):
    #conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='yaoyj11 '")
    #cur = conn.cursor()  
    #engine = create_engine('postgresql+psycopg2://postgres:yaoyj11 @localhost/postgres', poolclass=NullPool)
    
    # local machine usaid
    conn = psycopg2.connect(host="localhost",database="postgres", user="postgres") # connect to local host
    cur = conn.cursor() 
    engine = create_engine('postgresql+psycopg2://localhost/postgres') 

    df = get_data('train_prop_hollande_pr12t1_an','outcome')        
    holdout_df =get_data('hold_prop_hollande_pr12t1_an','outcome') 
    dictionary = pickle.load(open('data/dictionary' , "rb"))
    
    table_name = 'real_exp'
    num_covs = np.size(df,1) - 8 # index, iv, treat, outcome, zr, zd, match 
    cur.execute('drop table if exists {}'.format(table_name))
    conn.commit()

    
    print("copying data to sql server")    
    df.to_sql(table_name, engine)
    print("copied ...")    

    res = run_db(cur, conn,table_name, holdout_df, num_covs, dictionary)

    LATE, std = get_LATE_and_CI(cur,table_name,df,res)

    print(LATE, std)
    
    # see data after matching
    # df_after_match = pd.read_sql("select * from real_exp", conn)
    # assign unique group id (rather defined by two columns)
    df_after_match_2 = assing_gr_id(conn)
    # get data for se calculation
    df_se = df_after_match_2[ ["iv","treated","group_id_overall","outcome"] ]
    # drop observations with no recorded outcome values
    df_se.drop(df_se.index[np.isnan(df_se).sum(1) > 0], axis=0, inplace=True)
    # terrible hack to find out strata in which there are no treated/untreated obs
    to_ex = df_se.groupby("group_id_overall").apply(lambda x: np.mean(x.ix[:, 0]) == 0 or np.mean(x.ix[:, 0]) == 1).index[df_se.groupby("group_id_overall").apply(lambda x: np.mean(x.ix[:, 0]) == 0 or np.mean(x.ix[:, 0]) == 1)]
    # drop those strata
    df_se.drop(df_se.index[df_se.ix[:, 2].apply(lambda x: x in to_ex)], axis=0, inplace=True)
    
    df_se.to_csv("Low_t1_holl.csv")
    G = df_se.ix[:, 2]
    Z = df_se.ix[:, 0]
    T = df_se.ix[:, 1]    
    Y = df_se.ix[:, 3] # it is for index of output variable
    
    lhat = lambda_hat(Y, T, Z, G)
    sd_se = np.sqrt(Var_lambda_hat(Y, T, Z, G))
    tstat = lhat / sd_se
    print "Outcome Variable: %s, \n LATE: %.5f, \n se: %.5f, \n t-stat = %.5f" \
              % ("outcome", lhat, sd_se, tstat)

    df_after_match_2 = assing_gr_id(conn)
    df2 = df_after_match_2['matched']
    df2.columns = ["matched1","matched2"]
    df2['matched1'].value_counts()
         
    # run se calculation
    analyze_group = 0
    if analyze_group == 1 :
        pickle.dump(df_after_match_2, open('data/compare_groups/matched_train_prop_hollande_pr12t1_an', 'wb')) # cov "0" is strata
    
    return LATE

if __name__ == '__main__':

    # home directory 
    os.chdir("/Users/musaidawan/Dropbox/Duke/Projects/IV FLAME/Application/5 mins vote/replication_with_flame")
    
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn
    warnings.simplefilter(action='ignore', category=FutureWarning)

    res = run("full", 'outcome')
    
    
    
    
    
   
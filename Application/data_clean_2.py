'''
 - different from data_clean in following sense: 
         1 - train data covariates are categorical not dummies
         2 - same for holdout data
         3 - project_hold_data in IV-Flame to creates dummies in holdout for covs to match on

'''

import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split # use to split data
import os

# home directory 
os.chdir("/Users/musaidawan/Dropbox/Duke/Projects/IV FLAME/Application/5 mins vote/replication_with_flame")

#read in data
df = pd.read_stata('analysis.dta')
df = df.reset_index()

#select full control list

full_controls = ["territory", "prop_leftabstention_an", "nb_registered_mun", "prop_turnout_pr07t1_an",  
                 "nb_registered_pr12t1_an", "population", "share_men", "share_0014", "share_1529", 
                 "share_3044", "share_4559", "share_6074","share_75p", "share_unemployed", 
                 "share_working_pop", "population_delta", "share_men_delta", "share_0014_delta","share_1529_delta", 
                 "share_3044_delta", "share_4559_delta", "share_6074_delta","share_75p_delta","share_unemployed_delta", 
                 "share_working_pop_delta"]

#full_controls = full_controls[1:]

outcomes =["prop_turnout_pr12t1_an", "prop_turnout_pr12t2_an", "prop_turnout_pr12t12_an", 
           "prop_hollande_pr12t1_an", "prop_hollande_pr12t2_an","prop_hollande_pr12t12_an"]

# remove rows with missing values of outcome
for outcome in outcomes:
		df = df.loc[~np.isnan(df[outcome])] 
            # remove rows for which any of the outcome is missing!

df = df.loc[~np.isnan(df["prop_turnout_pr07t1_an"])]  # dropping missing values of prop_turnout_pr07t1_an
print(df[full_controls].isnull().any())  # no col is missing!

# discretize continuous variables            
for cov in full_controls: # bin cont covs
    if cov in ["territory"] :
        continue
    if cov in ["prop_leftabstention_an"] :
        df[cov] = pd.cut(df[cov], bins = 10 , labels = ["1", "2", "3", "4","5","6","7","8","9","10"] ) # create 5 bins for each    
    else :
        df[cov] = pd.cut(df[cov], bins = 5 , labels = ["1", "2", "3", "4","5"] ) # create 5 bins for each

df_cov = df[["index"]+full_controls]

# rename columns for iv and treated
df.rename(index=str, columns={"treatment": "iv", "allocated": "treated"},inplace=True)


# Appraoch 2 to sampling
    # train and holdout stratas
    #df_strat = df.drop_duplicates(['stratum_identifier']) # drop duplicates
    #df_strat = df_strat['stratum_identifier'].to_frame()
    #sample_size = int(0.40*np.size(df_strat,0))
    #
    #df_sample_strat = df_strat.sample(n=sample_size, random_state=2019)
    #df_sample_strat["holdout_strat"] = 1
    #
    #df_strat = pd.merge(df_strat, df_sample_strat, how='left', on=['stratum_identifier']) 
    #df_strat["holdout_strat"] = df_strat["holdout_strat"].fillna(0)   
    #
    ## train and holdout index
    #df_sample_index = pd.merge(df[['index','stratum_identifier']], df_strat, how='left', on=['stratum_identifier']) 
    #df_sample_index= df_sample_index[['index','holdout_strat']]
    #df_sample_index = df_sample_index.rename(columns = {'holdout_strat':'sample'}) # change col name

# Approach 1 to sampling
    # index values of random sample
    #sample_size = int(0.40*np.size(df,0))
    #df_sample_index = df.sample(n=sample_size, random_state=2019)
    #df_sample_index = df_sample_index["index"].to_frame()
    #df_sample_index["sample"] = 1

#select and save data
for outcome in outcomes :
    
    i_t_y = ["index","iv","treated",outcome]
    df_i_t_y = df[i_t_y]
    df_i_t_y.columns = ["index","iv","treated","outcome"]
    
    # rename covariates in df_cov
    q = np.size(df_cov, 1) # number of feature columns
    ls = range( q  - 1) # -1 for index col
    ls = map(str, ls) 
    
    
    # df_cov
    df_cov = df[["index"]+full_controls]
    
    # dictionary for renames 
    covs = list(df_cov.columns)
    covs = covs[1:]
    dictionary = dict(zip(ls, covs))
    
    df_cov.columns = ["index"] + ls

    # df_train 
    df_train_0 = pd.merge(df_cov,df_i_t_y,on='index',how='inner')
    #df_train_0 = pd.merge(df_train_0,df_sample_index,on='index',how='left') # sampling
    #df_train_0["sample"] = df_train_0["sample"].fillna(0)
    #df_train = df_train_0[ df_train_0["sample"] == 0  ] 
    #df_train = df_train.drop(['sample'], axis=1)
    
    train, test = train_test_split(df_train_0, test_size=0.15, random_state = 123, stratify =df_train_0[["iv","treated"]] )
    
    #X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)
    
    # df_hold (random sample of df_train)
    #df_hold = df_train_0[ df_train_0["sample"] == 1  ] 
    #df_hold = df_hold.drop(['sample'], axis=1)
    
    # check for missing values
    df_train = train
    df_hold = test
    print(df_train.isnull().any())
    print(df_hold.isnull().any())
    
    pickle.dump(df_train, open('data/train_'+ outcome, 'wb')) # cov "0" is strata
    pickle.dump(df_hold, open('data/hold_'+ outcome, 'wb')) # cov "0" is strata
    pickle.dump(dictionary, open('data/dictionary', 'wb'))         


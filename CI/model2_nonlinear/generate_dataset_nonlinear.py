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
from sqlalchemy.pool import NullPool

import sql
from sklearn import feature_selection

from sklearn import linear_model
import statsmodels.formula.api as sm
from statsmodels.stats import anova
import pylab as pl
from multiprocessing import Pool
from functools import partial
import warnings
import pysal
from pysal.spreg.twosls import TSLS
import random

def construct_sec_order(arr): 
    second_order_feature = []
    num_cov_sec = len(arr[0])
    for a in arr:
        tmp = []
        for i in range(num_cov_sec):
            for j in range(i+1, num_cov_sec):
                tmp.append( a[i] * a[j] )
        second_order_feature.append(tmp)
        
    return np.array(second_order_feature)

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

    """
    threshold = 1.0
    Dij = np.asarray([0 if e < threshold else 1 for e in dij[:,0]]).reshape((num_treated + num_control,1))
    Dij = dij
    """

    threshold1 = 0.3
    threshold2 = 0.6
    threshold3 = 1.0
    Dij = np.asarray([0 if e < threshold1 else 1 if e < threshold2 else 2 if e < threshold3 else 3 for e in dij[:,0]]).reshape((num_treated + num_control,1))

    gamma = get_gamma(dim,0.5,5)
    Rij = np.add(beta * Dij, np.matmul(x, gamma).reshape((num_treated + num_control,1)))
    second_order = 10 * construct_sec_order(x[:,:5]).sum(axis = 1)
    Rij = np.add(Rij,second_order.reshape((num_treated + num_control,1)))
    Rij = np.add(Rij, epsilon_ksi[:,0].reshape([num_treated + num_control,1]))     

    df = pd.DataFrame(np.concatenate([x, z, Dij, Rij], axis = 1))
    #df.columns = df.columns.astype(str)
    start_index = dim
    df.rename(columns = {start_index:"iv"}, inplace = True)
    df['iv'] = df['iv'].astype('int64')  
    df.rename(columns = {start_index+1:"treated"}, inplace = True)
    df['treated'] = df['treated'].astype('int64') 
    df.rename(columns = {start_index+2:"outcome"}, inplace = True)

    df['zr'] = df['iv'] * df['outcome']
    df['zd'] = df['iv'] * df['treated']
    df['matched'] = 0


    df.reset_index()
    df['index'] = df.index

    return df,x,z,Dij,Rij
    
def run(pi):
    df_sub = []
    np.random.seed()

    for i in range(1):
        print(pi,i)
        num_treated = 1500
        num_control = 1500
        cov = 0.5
        df,x,z,d,r = data_generation_dense_2(cov,num_control, num_treated, 8, 2, pi)
        df_sub.append(df)
        np.savetxt("data/x_nonlinear_15000_multilevel_"+str(pi)+"_"+str(i)+".txt",x,fmt="%d")
        np.savetxt("data/z_nonlinear_15000_multilevel_"+str(pi)+"_"+str(i)+".txt",z,fmt="%d")
        np.savetxt("data/d_nonlinear_15000_multilevel_"+str(pi)+"_"+str(i)+".txt",d,fmt="%.8f")
        np.savetxt("data/r_nonlinear_15000_multilevel_"+str(pi)+"_"+str(i)+".txt",r,fmt="%.8f")
    
    return pi,df_sub

if __name__ == '__main__':
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    pi_array = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    #pi_array = [0.5]
    
    with Pool(min(10, len(pi_array))) as p:
        drop_results = p.map(run, pi_array)
        for pi, df_sub in drop_results:
            pickle.dump(df_sub, open('data/df_nonlinear_15000_multilevel_'+str(pi), 'wb'))
           

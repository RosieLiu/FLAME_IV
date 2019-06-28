#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:52:31 2019

@author: musaidawan
"""

import numpy as np
import pandas as pd
import os
from sqlalchemy import create_engine
import psycopg2



def update_matched(cur, conn, covs_matched_on, db_name, level):  
    covs_matched_on = set(covs_matched_on)

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

    conn.commit()

    return 

if __name__ == '__main__':


    # test dataframe to test new matching algorithm + group ids    
    d = {'1': [1,1,2,1],
         '2': [1,1,1,2],
         "iv":[1,0,0,0],
         'matched': [0,0,0,0],
         'group_id' : [-99,-99,-99,-99]}
    
    df = pd.DataFrame(d)
    
    
    # connection and move df to serve
    
    conn = psycopg2.connect(host="localhost",database="postgres", user="postgres") # connect to local host
    cur = conn.cursor() 
    engine = create_engine('postgresql+psycopg2://localhost/postgres') 
        
    # move df to serve
    table_name = 'test'
    cur.execute('drop table if exists {}'.format(table_name))
    conn.commit()
    df.to_sql(table_name, engine)    

    # inputs to match function
    covs_matched_on = ["1", "2"]
    level = 1
    

    # check data before edits    
    df_test_before = pd.read_sql("select * from test ", conn)

    # check grouping result
    query = ''' SELECT {0},
        ROW_NUMBER() OVER (ORDER BY {0}) as "group_id"        
        FROM {3}
        where "matched"=0
        group by {0}
        Having sum("iv") > 0 and sum("iv") < count(*)'''.format(','.join(['"{0}"'.format(v) for v in covs_matched_on]), #0
                   ','.join(['{1}."{0}"'.format(v, table_name) for v in covs_matched_on]),
                   ' AND '.join([ '{1}."{0}"=temp."{0}"'.format(v, "T1") for v in covs_matched_on ]),
                   table_name, #3
                   level #4
                  )
    df_grp = pd.read_sql(query, conn)


    # apply matching update query
    update_matched(cur, conn, covs_matched_on, table_name, level)

    # chech data after query
    df_test_after = pd.read_sql("select * from test ", conn)
    
    # to assign group id by looking at two columns
    d = { 'matched': [1,1,1,1,2,2,2,2],
         'group_id' : [1,1,2,2,1,1,4,4]}
    
    df = pd.DataFrame(d)





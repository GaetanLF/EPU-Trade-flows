#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:38:42 2022

@author: gaetanlefloch
"""

import pandas as pd
import numpy as np
import sqldf
import gme

df = pd.read_stata('data.dta')
df['log_distance'] = np.log10(df['dist'])
EPU = pd.read_csv('EPU_means.csv',sep=';')

countries_list = ['France','Australia','Brazil','Canada',
                                'Germany','India','Italy','Mexico','Rep. of Korea',
                                'Russian Federation','United Kingdom','USA',
                                'Chile','China','Colombia','Greece','Ireland',
                                'Japan','Netherlands','Spain',
                                'Sweden']

df = df.loc[df['cname_i'].isin(countries_list)]
df = df.loc[df['cname_j'].isin(countries_list)]
df = df.loc[df['year'].isin([2000,2002,2004,2006,2008,2010,2012,
                            2014,2016,2018])]

query1 = '''
    SELECT df.*, EPU.EPU as EPU_i
        FROM df
        LEFT JOIN EPU ON
        df.cname_i = EPU.Country AND
        df.year = EPU.Year;
'''

query2 = '''
    SELECT df.*, EPU.EPU as EPU_j
        FROM df
        LEFT JOIN EPU ON
        df.cname_j = EPU.Country AND
        df.year = EPU.Year;
'''

df = sqldf.run(query1)
df = sqldf.run(query2)
df['EPU_combined'] = df['EPU_i']+df['EPU_j']

gme_data = gme.EstimationData(data_frame=df,
                              imp_var_name='cname_i',
                              exp_var_name='cname_j',
                              trade_var_name='trade',
                              year_var_name='year')

fixed_effects_model  = gme.EstimationModel(estimation_data = gme_data,
                                 lhs_var = 'trade',
                                 rhs_var = ['gdp_i',
                                            'gdp_j',
                                            'log_distance',
                                            'comrelig',
                                            'contig',
                                            'EPU_combined'],
                                 fixed_effects=['cname_i'])
estimation = fixed_effects_model.estimate()

results = estimation['all']
results.summary()

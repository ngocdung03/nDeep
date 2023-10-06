import pandas as pd
import numpy as np
from sklearn import linear_model
import os
import sys
dir, filename = os.path.split(os.path.abspath(__file__))
HOME = os.path.split(dir)[0]

def _is_upper(string):
    return string == string.upper()

def count_cancer(name, df=None):
    if type(df) == pd.DataFrame:
        tmp = 's' + name + '_b'
        if _is_upper(name):
            return len(df[(df[name] >= 1) & (df[tmp.upper()] > 0)])
        else:
            return len(df[(df[name] >= 1) & (df[tmp] > 0)])
    return

def make_2nd_variable(name, df=None):
    if type(df) == pd.DataFrame:
        if name == 'pp':
            df['PP'] = df['SBP_B'] - df['DBP_B']
            return df
        elif name == 'ccr':
            df['CCR'] =  (140 - df['AGE_B']) * df['WT_B'] / (df['CREAT_B'] * 72)
            df['CCR'].where(df['SEX1'] == 2, df['CCR'] * 0.85)
            return df
        elif name == 'egfr':
            df['eGFR'] = 186.3 * (df['CREAT_B']**-1.154) * (df['AGE_B']**-0.203)
            df['eGFR'].where(df['SEX1'] == 2, df['eGFR'] * 0.742)
            return df
        
    return df

def get_features(filename):
    with open(filename, 'r') as f:
        return [x.strip('\n') for x in f.readlines() if '#' not in x]
    
def get_cancers(filename):
    with open(filename, 'r') as f:
        c = [x.strip('\n') for x in f.readlines()]
    return c, ['S'+x+'_B' if _is_upper(x) else 's'+x+'_b' for x in c ]

def _imputation_by_sex(df, refer, bysex=False, method='mean'):
    if bysex:
        mean = pd.core.groupby.DataFrameGroupBy.mean
        median = pd.core.groupby.DataFrameGroupBy.median
        try:
            refer = eval(method)(refer.groupby(['SEX1']))
        except:
            raise Exception('Wrong method!')
        refer = eval(method)(refer.groupby(['SEX1']))
        refer = refer.fillna(refer.describe().loc['mean'])
        return pd.concat([df[df['SEX1'] == k].fillna(refer.loc[k]) for k in df['SEX1'].unique()]).sort_index()
    mean = pd.DataFrame.mean
    median = pd.DataFrame.median
    return df.fillna(eval(method)(refer))
    
    
def imputation(df, refer=None, method='mean', bysex=False):
    if method not in ['mean', 'median', 'regression']:
        raise TypeError('Wrong method!')
    
    if method == 'mean':
        refer = df.copy(deep=True) if refer is None else refer
        return _imputation_by_sex(df, refer, bysex=bysex)
    elif method == 'median':
        refer = df.copy(deep=True) if refer is None else refer
        return _imputation_by_sex(df, refer, bysex=bysex, method=method)
    elif method == 'regression':
        df_reg = df.copy(deep=True)
        feature = list(df.columns)
        for y in feature[2:]:
            notnull = df_reg[['AGE_B', 'SEX1', y]].dropna()
            if len(notnull) > 0:
                if refer != None:
                    predict = refer.predict(df_reg[['AGE_B', 'SEX1']])
                    df_reg[y] = np.where(df_reg[y].isna(), predict, df_reg[y])
                else:
                    model = linear_model.LinearRegression()
                    model.fit(X = notnull[['AGE_B', 'SEX1']], y = notnull[y])
                    predict = model.predict(df_reg[['AGE_B', 'SEX1']])
                    df_reg[y] = np.where(df_reg[y].isna(), predict, df_reg[y])
        if refer != None:
            return df_reg
        return df_reg, model
    return

def imputation2(data, method = 'mean', feature=None):
    data_reg = data.copy(deep=True)
    if method == 'mean':
        return data_reg.fillna(data_reg.mean())
    elif method == 'median':
        return data_reg.fillna(data_reg.median())
    elif method == 'regression':
        for y in feature[2:]:
            notnull = data_reg[['AGE_B', 'SEX1', y]].dropna()
            model = linear_model.LinearRegression()
            model.fit(X = notnull[['AGE_B', 'SEX1']], y = notnull[y])
            predict = model.predict(data_reg[['AGE_B', 'SEX1']])
            data_reg[y] = np.where(data_reg[y].isna(), predict, data_reg[y])
        return data_reg
    

def imputation_by_age(df, method='mean', interval=10):
    ages = range(0, 90, interval)
    tmp = []
    for age in ages:
        d = df[(df.AGE_B >=age) & (df.AGE_B < age+interval)] 
        if age == 0 or age==70:
            a = d.copy(deep=True)
        elif age == 10 or age == 80:
            d = pd.concat([a, d], axis=0)
            tmp.append(imputation(d, method=method))
        else:
            tmp.append(imputation(d, method=method))
    return pd.concat(tmp, axis=0)

def get_not_na(df):
    return df[[df.columns[i] for i, x in enumerate(df.isna().sum(axis=0)) if x == 0]]
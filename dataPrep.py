import numpy as np
import pandas as pd
import os
from urllib.request import urlopen
import json

data_dir = './Data'

# Base data
def df_type_trans(df):
    df['Quote_dt'] = pd.to_datetime(df['Quote_dt'], format = '%Y-%m-%d')
    df['zip'] = df['zip'].astype('Int64').astype('object')
    df['Agent_cd'] = df['Agent_cd'].astype('Int64').astype('object')
    df['CAT_zone'] = df['CAT_zone'].astype('Int64')
    try:
        df['high_education_ind'] = df['high_education_ind'].astype('Int64')
    except:
        None
    return df

def load_df(train_test_split=True):
    df = pd.read_csv(os.path.join(data_dir, 'analytical_df.csv'))
    df = df_type_trans(df)  # fix dtypes
    # create time based features
    df = df.assign(
        dayofweek = lambda x: x.Quote_dt.dt.dayofweek,
        month = lambda x: x.Quote_dt.dt.month,
        quarter = lambda x: x.Quote_dt.dt.quarter
    )
    return (
        df[lambda x: x.split == 'Train'].drop(['split'], axis=1),
        df[lambda x: x.split == 'Test'].drop(['split'], axis=1)
    ) if train_test_split else df

def get_policy_df():
    policy = pd.read_csv(os.path.join(data_dir, 'policies.csv'))
    policy['quoted_amt'] = policy.quoted_amt.str.replace(r'\D+', '', regex=True).astype('float')
    policy = df_type_trans(policy)
    policy = policy.assign(
        dayofweek = lambda x: x.Quote_dt.dt.dayofweek,
        month = lambda x: x.Quote_dt.dt.month,
        quarter = lambda x: x.Quote_dt.dt.quarter
    )
    return policy[lambda x: x.split == 'Train'].drop(['split'], axis=1)

# Time series data
def get_ts_data(train_test_split=True, get_holiday=False):
    # do not use test data in time series analysis
    df,_ = load_df(train_test_split=train_test_split)
    if get_holiday:
        import holidays
        df['holiday'] = df['Quote_dt'].apply(lambda x: 0 if holidays.US().get(x) is None else 1)
    df = df.groupby('policy_id', as_index= False).first()
    return df

def query_ts_data(resample='M', query=None):
    '''resample: specifies the resample rule; query: to slice df'''
    if query is not None:
        df = get_ts_data().query(query)
    else:
        df = get_ts_data()
    query_df = df.set_index('Quote_dt')['convert_ind'].resample(resample).apply(['sum','count']).assign(cov_rate = lambda x: x['sum']/x['count'])
    return query_df

# Sales analysis data
def get_revenue_df():
    '''returns the revenue_df and counties geojson file for county based map plot'''
    policy = get_policy_df()
    # get geojson data
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    # county_fips = {"county_name": "fips_id"}
    county_fips = {}
    for i, geoinfo in enumerate(counties["features"]):
        county_fips[geoinfo['properties']['NAME']] = geoinfo['id']
    revenue_df = pd.merge(
        policy.groupby('county_name', as_index=False).first()[['state_id','county_name']],
        policy.assign(
            revenue = lambda x: x['quoted_amt']*x['convert_ind']
            ).groupby('county_name', as_index=False).sum().assign(
                fips = lambda x: x['county_name'].map(county_fips)
                )[['county_name', 'fips', 'revenue']],
        on='county_name'
        )
    return revenue_df, counties


# Utils
def get_conversion_rate(df, variables=['var1','var2'], pivot=False):
    # get num of convert and total num of policy
    var_count = df.groupby(variables)['convert_ind'].aggregate(['sum', 'count']).reset_index()
    # get conversion rate
    var_count['conversion_rate'] = var_count['sum'] / var_count['count']
    var_count = var_count.rename(columns={'sum':'num_converted','count':'total'})
    # create pivot table
    if (len(variables) != 1) & pivot:
        var_pivot = var_count.pivot(
            index=variables[0],
            columns=variables[1],
            values='conversion_rate'
            ).fillna(0)
    elif len(variables) == 1:
        pivot = False
        print('Only one variable passed')
    else:
        pivot = False
    return var_pivot if pivot else var_count

## Testing
# print(get_revenue_df()[0])
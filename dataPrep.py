import numpy as np
import pandas as pd
import os
from urllib.request import urlopen
import json
import streamlit as st

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

@st.cache(ttl=24*3600)
def load_df(train_test_split=True):
    """join policy and driver dataset, each record represents a driver on the policy"""
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

def fill_missing(df):
    """the dtypes of the predictors should match their meaning"""
    # initialize a new df
    train_fill = pd.DataFrame()
    # find the numeric variables
    num_var = df.select_dtypes(include=['float']).columns
    # fill the categorical vars with missing or 0 (label encoded or discrete vars)
    for i, predictor in enumerate(df.drop(columns=num_var)):
        try:
            train_fill[predictor] = df[predictor].fillna('Missing')
        except:
            train_fill[predictor] = df[predictor].fillna(0)
    # fill the numeric vars with mean
    for i, predictor in enumerate(df[num_var]):
        train_fill[predictor] = df[predictor].fillna(df[predictor].mean())
    return train_fill

@st.cache(ttl=36*3600)
def get_policy_df():
    """return training set loaded from the original policy dataset with preprocessing"""
    policy = pd.read_csv(os.path.join(data_dir, 'policies.csv'), index_col=0)
    policy['quoted_amt'] = policy.quoted_amt.str.replace(r'\D+', '', regex=True).astype('float')
    policy = df_type_trans(policy)
    policy = policy.assign(
        dayofweek = lambda x: x.Quote_dt.dt.dayofweek,
        month = lambda x: x.Quote_dt.dt.month,
        quarter = lambda x: x.Quote_dt.dt.quarter
    )
    return policy[lambda x: x.split == 'Train'].drop(['split'], axis=1)

# Time series data
@st.cache(ttl=24*3600)
def get_ts_data(train_test_split=True, get_holiday=False):
    # do not use test data in time series analysis
    df,_ = load_df(train_test_split=train_test_split)
    if get_holiday:
        import holidays
        df['holiday'] = df['Quote_dt'].apply(lambda x: 0 if holidays.US().get(x) is None else 1)
    df = df.groupby('policy_id', as_index= False).first()
    return df

@st.cache(ttl=24*3600)
def query_ts_data(resample='M', query=None):
    '''resample: specifies the resample rule; query: to slice df'''
    if query is not None:
        df = get_ts_data().query(query)
    else:
        df = get_ts_data()
    query_df = df.set_index('Quote_dt')['convert_ind'].resample(resample).apply(['sum','count']).assign(cov_rate = lambda x: x['sum']/x['count'])
    return query_df

# Customer data
@st.cache(ttl=24*3600)
def plot_family_status():
    vars = ['policy_id', 'number_drivers', 'gender', 'living_status', 'high_education_ind', 'age', 'convert_ind']
    df,_ = load_df()
    train_fill = fill_missing(df)
    family_df = train_fill[vars]
    # get family status label
    adults = family_df.iloc[np.where(family_df.living_status != 'dependent')].groupby('policy_id', as_index=False)['convert_ind'].count().rename(columns={'convert_ind': 'adult_count'})
    children = family_df.iloc[np.where(family_df.living_status == 'dependent')].groupby('policy_id', as_index=False)['convert_ind'].count().rename(columns={'convert_ind': 'children_count'})
    family_df = pd.merge(family_df, adults, on='policy_id', how='left')
    family_df['adult_count'] = family_df['adult_count'].fillna(0.0)  # there are certain families have no adult
    family_df = pd.merge(family_df, children, on='policy_id', how='left')
    family_df['children_count'] = family_df['children_count'].fillna(0.0)  # there are certain families have no children  
    family_df = family_df.assign(
        family_status=lambda x: np.where(
            (x.adult_count == 2) & (x.children_count == 0), "Couple", np.where(
                (x.adult_count == 1) & (x.children_count > 0), "Single Parent", np.where(
                    (x.adult_count == 1) & (x.children_count == 0), "Single Adult", np.where(
                        x.adult_count == 0, "Dependent Child", "Family"))))
    )
    try:
        # children living status is missing, however still considered a couple
        policy_26680_family = family_df.iloc[np.where(family_df.policy_id=="policy_26680")].replace("Family", "Couple")
        family_df[lambda x: x.policy_id=="policy_26680"] = policy_26680_family
    except:
        None
    return family_df

# Sales analysis data
@st.cache(ttl=24*3600)
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



# Testing
if __name__ == "__main__":
    ts = get_ts_data()
    print(
        ts.head(), "\n", ts.shape
    )
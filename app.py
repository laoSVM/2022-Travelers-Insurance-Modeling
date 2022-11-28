import pickle
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import lightgbm as lgb
from dataPrep import *
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from PIL import Image
import statsmodels.stats.proportion as sp


# st.set_option("browser.gatherUsageStats", False)
PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"wide"}
st.set_page_config(**PAGE_CONFIG)
benchmark_file = "benchmark_2.pkl"

# cache the model
@st.cache
def get_model(benchmark_file):
    with open(benchmark_file, 'rb') as f:
        benchmark = pickle.load(f)
    return benchmark["model"], list(benchmark["dataset"].drop(['policy_id', 'split', 'convert_ind'], 1).columns)

def preprocess(df):
    df['discount'] = int(df['discount'])
    columns = df.select_dtypes(include=["object_"]).columns
    lbEncode = {'state_id': {'AL': 0, 'CT': 1, 'FL': 2, 'GA': 3, 'MN': 4, 'NJ': 5, 'NY': 6, 'WI': 7}, 'discount': {'No': 0, 'Yes': 1}, 'Prior_carrier_grp': {'Carrier_1': 0, 'Carrier_2': 1, 'Carrier_3': 2, 'Carrier_4': 3, 'Carrier_5': 4, 'Carrier_6': 5, 'Carrier_7': 6, 'Carrier_8': 7, 'Other': 8}}
    for cat_col in columns:
        df[cat_col] = df[cat_col].map(lbEncode[cat_col])
    return df

def make_prediction(df, model) -> float:
    return model.predict_proba(df)[:,1]

clf, columns = get_model(benchmark_file)
predictors = pd.DataFrame(np.zeros(len(columns)).reshape(1,-1), columns=columns)

def main():
    st.title("Streamlit App for 2022 Travelers") # main title
    # side panel for user input
    with st.sidebar:
        st.subheader("Inputs")
        st.markdown("**Personal Information**")
        predictors['quoted_amt'] = st.number_input(
            "Quoted Amt:",
            0.0,9999999.0,5876.0,
            step=1.0
        )
        predictors['drivers_age'] = st.number_input(
            "Customer age:",
            0.0,100.0,40.0,
            step=1.0
        )
        predictors['credit_score'] = st.number_input(
            "Customer credit score:",
            300.0,850.0,642.0,
            step=1.0
        )
        predictors['total_number_veh'] = st.number_input(
            "Total number vehicles on the policy:",
            1,10,3,
            step=1
        )
        predictors['Prior_carrier_grp'] = st.selectbox(
            "Prior carrier group:",
            ['Carrier_1','Carrier_2','Carrier_3','Carrier_4','Carrier_5','Carrier_6','Carrier_7','Carrier_8','other']
        )
        predictors['discount'] = st.checkbox("Discount applied")

        st.markdown("---")
        # area for locational info such as state and cat_zone
        st.markdown("**Location Information**")
        predictors['state_id'] = st.selectbox(
            'Pick the state:',
            ['NY', 'FL', 'NJ', 'CT', 'MN', 'WI', 'AL', 'GA'])
        predictors['CAT_zone'] = st.select_slider(
            'CAT_zone', 
            [1,2,3,4,5])
        submitted = st.button('Submit')
    
    # main panel
    with st.expander('About this app'):
        st.markdown('This app shows the **conversion probability** based on the information you provide.')
        st.write('😊Happy Coding.')
    tsTab, customerTab, salesTab, predictionTab = st.tabs(["Time Series", "Customer Group", "Marketing & Sales", "Prediction"])

    with tsTab:
        st.info("We have observed a drop in the amount of quotes issued, as well as conversion rates.")
        # prepare data
        df = get_ts_data()
        ts_trend = df.set_index('Quote_dt')['convert_ind'].resample('Q').apply(['sum','count']).assign(cov_rate = lambda x: x['sum']/x['count'])
        # plot bar chart
        fig = px.bar(ts_trend, x=ts_trend.index, y='count',
            color='cov_rate',
            color_continuous_scale='ice',
            labels={
                'Quote_dt': 'Quote issued date',
                'count': 'Num of quotes',
                'cov_rate': 'Conversion'},
            title='Quote Trends (by Quarter)'
        )
        fig.update_layout(
            title={
                'y':0.9,
                'x':0.5,})
        st.plotly_chart(fig)

        # assumptions
        tsQuests = [
            'How does conversion rates change over the years?',
            'Does conversion rates have seasonality?',
            'More analysis inprogress...'
        ]
        tsQuest = st.selectbox('More detailed analysis', tsQuests)

        if tsQuest == tsQuests[0]:
            # if not specified: whole time series
            # start, end: datetime.date object
            start, end = st.slider(
                "Pick a date range of interest:",
                value=(date(2015,1,1), date(2018,12,31)),
                key='time_range')
            left, right = st.columns([1,4])
            with left:
                df = query_ts_data(resample='M')
                current_cov = df[lambda x: (
                    (x.index.year == end.year) &
                    (x.index.month == end.month)
                )]
                prev_month = pd.to_datetime(end) - pd.DateOffset(month=1)
                prev_cov = df[lambda x: (
                    (x.index.year == prev_month.year) &
                    (x.index.month == prev_month.month)
                )]
                st.metric('Conversion', f"{(current_cov['cov_rate'].values[0]):.2%}", f"{( (current_cov['cov_rate'].values[0] - prev_cov['cov_rate'].values[0]) / prev_cov['cov_rate'].values[0]) :.2%}")
                st.metric('Num Quotes', current_cov['count'], f"{( (current_cov['count'].values[0] - prev_cov['count'].values[0]) / prev_cov['count'].values[0]):.2%}")
            with right:
                start_f = start.strftime('%Y%m%d')
                end_f = end.strftime('%Y%m%d')
                # if specified time range: cal based on time range
                df = query_ts_data(resample='M', query=f'Quote_dt >={start_f} and Quote_dt <= {end_f}')
                fig = px.line(df, x=df.index, y='cov_rate',
                    labels={
                        'Quote_dt': 'Quote issued date',
                        'cov_rate': 'Conversion'
                    })
                st.plotly_chart(fig)
            
        if tsQuest == tsQuests[1]:
            st.subheader("We do not observe apparent autocorrelation and seasonality with conversion rates.")
            st.image(Image.open("./Image/partial correlation.png"))
            st.image(Image.open("./Image/seasonal decompose.png"))

        if tsQuest == tsQuests[2]:
            pass

    with customerTab:
        pass

    with salesTab:
        st.write("A few sales report.")
        salesQuests = [
            'Does providing discount increase conversion? -- A/B Test',
            'More analysis inprogress...'
        ]
        salesQuest = st.selectbox('More detailed analysis', salesQuests)

        if salesQuest == salesQuests[0]:
            # Prepare data for hypothesis testing
            policy = get_policy_df()
            discount_df = pd.merge(
                policy[policy['convert_ind']==0].groupby(['discount'], as_index=False)['policy_id'].count().rename(columns={'policy_id': "Not converted"}),
                policy[policy['convert_ind']==1].groupby(['discount'], as_index=False)['policy_id'].count().rename(columns={'policy_id': "Converted"}),
                on='discount'
            ).assign(sample_size = lambda x: x.sum(1))
            fig = px.bar(
                discount_df, x=['Not converted', 'Converted'], y='discount',
                orientation ='h')
            st.plotly_chart(fig)
            # Prepare experiment
            n_control = discount_df.sum(1)[0]
            n_test = discount_df.sum(1)[1]
            convert_control = discount_df.query('discount=="No"')['Converted'].values[0]
            convert_test = discount_df.query('discount=="Yes"')['Converted'].values[0]
            z_score, p_value = sp.proportions_ztest([convert_control, convert_test], [n_control, n_test], alternative='smaller')
            left, right = st.columns([2,4])
            with left:
                st.latex(r"H_0: P_{No}=P_{Yes}")
                st.latex(r"H_1: P_{No}<P_{Yes}")
                result = {
                    "Treatment": "Discount",
                    "Control Group Size": n_control,
                    "Treatment Group Size": n_test,
                    "Control Group Convert": convert_control,
                    "Treatment Group Convert": convert_test,
                    "p-value": round(p_value, 4)
                }
                st.dataframe(pd.DataFrame(result, index=['Result']).T)
                st.markdown("According to the result, **p-value<0.05**. Therefore we reject the null hypothesis. **Giving discounts to customers does have a positive effect** in conversion.")
            with right:
                fig = px.pie(discount_df,
                            values='sample_size', names='discount')
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    title='Sample Proportions',
                    legend_title="Discount")
                st.plotly_chart(fig)
            # Plot line charts indicating conversion change
            discount_No = query_ts_data(resample='M', query='discount=="No"').reset_index(drop=False)
            discount_Yes = query_ts_data(resample='M', query='discount=="Yes"').reset_index(drop=False)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=discount_No["Quote_dt"], y=discount_No["cov_rate"],
                                mode='lines', name='No'))
            fig.add_trace(go.Scatter(x=discount_Yes["Quote_dt"], y=discount_Yes["cov_rate"],
                                mode='lines', name='Yes'))
            fig.update_layout(
                title='Discount & Conversion Rate',
                xaxis_title='Time',
                yaxis_title='Conversion Rate',
                legend_title="Discount")
            st.plotly_chart(fig)

    with predictionTab:
        if submitted:
            # show the df for test purpose
            st.dataframe(predictors)
            predictorsTrans = preprocess(predictors)
            st.dataframe(predictorsTrans)
            # make prediction
            st.metric('Conversion Rate', make_prediction(predictorsTrans, clf))
        else:
            st.info("Please submit the customer information", icon="👈")
     

if __name__ == '__main__':
    main()
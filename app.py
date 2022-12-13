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
from scipy.stats import chi2_contingency


# st.set_option("browser.gatherUsageStats", False)
PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"wide"}
st.set_page_config(**PAGE_CONFIG)
hide_st_style = """
<style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
#header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
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
    st.title("Auto Quote Conversion Dashboard") # main title
    # side panel for user input
    with st.sidebar:
        st.subheader("User Inputs")
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
            'How does conversion rate change over the years?',
            'Does conversion rate have seasonality?',
            'Market share comparison across multiple areas.',
            'More analysis inprogress...'
        ]
        tsQuest = st.selectbox('More detailed analysis', tsQuests)

        if tsQuest == tsQuests[0]:
            st.info("In general, conversion rate follows a steady drop over the years. It fluctuates and reached its bottom around mid 2017.")
            # if not specified: whole time series
            # start, end: datetime.date object
            start, end = st.slider(
                "Pick a date range of interest:",
                value=(date(2015,1,1), date(2018,12,31)),
                key='time_range')
            left, right = st.columns([1,4])
            with left:
                df = query_ts_data(resample='M')
                # get the conversion rates from the month where the end point of the slider lies
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
            policy = get_policy_df()
            states_customer = policy.set_index('Quote_dt').groupby([pd.Grouper(freq='M'), 'state_id'])['convert_ind'].agg(['sum', 'count']).reset_index(drop=False)
            st.info("**NY, FL, NJ** are the main markets compared with other states. Starting from the beginning of 2017, the quotes from NY, FL, NJ droped seriously. Other states also experienced a small drop.")
            fig = px.line(
                states_customer, x='Quote_dt', y='count', color='state_id',
                labels={
                    'Quote_dt': 'Quote issued date',
                    'count': 'Num of quotes'
                }
            )
            st.plotly_chart(fig)

        if tsQuest == tsQuests[3]:
            pass

    with customerTab:
        st.write("Some analysis about customers")
        customerQuests = [
            'How does family size affect conversion?',
            'Are conversion rates different among groups?',
            'More analysis inprogress...'
        ]
        customerQuest = st.selectbox('More detailed analysis', customerQuests)

        if customerQuest == customerQuests[0]:
            agg_dict = {
                'policy_id':'count',
                'convert_ind':'first'}
            train, _ = load_df()
            family_size_df = train.groupby('policy_id', as_index=False).agg(agg_dict)[['policy_id','convert_ind']].rename(columns={'policy_id': 'family_size'})
            fs_cov = get_conversion_rate(family_size_df, ['family_size'])
            fig = px.bar(
                fs_cov, x=['conversion_rate'], y='family_size',
                orientation ='h')
            st.plotly_chart(fig)
        if customerQuest == customerQuests[1]:
            var_list = ['discount', 'Home_policy_ind', 'state_id', 'Prior_carrier_grp', 
                        'Cov_package_type', 'CAT_zone', 'number_drivers', 'num_loaned_veh', 
                        'num_owned_veh', 'num_leased_veh', 'total_number_veh', 'primary_parking']
            variables = st.multiselect(
                'Select variables that interest you the most:',
                var_list, ['discount', 'Cov_package_type'],
                max_selections=2)
            train, _ = load_df()
            cov_train = train.query('convert_ind==1')
            if len(variables)==1:
                cnt_tab = get_conversion_rate(train, [variables[0]], pivot=False)
                st.dataframe(cnt_tab)
                fig = px.bar(
                    cnt_tab, x=["total", "num_converted"], y=variables[0], orientation ='h',
                    labels={"conversion_rate": "Conversion Rate"},
                    hover_data=['conversion_rate'])
                st.plotly_chart(fig)
            elif len(variables)==2:
                cnt_tab = pd.crosstab(cov_train[variables[0]], cov_train[variables[1]])
                # render a heatmap showing conversion table
                fig = px.imshow(cnt_tab, color_continuous_scale='ice', text_auto=True,
                labels=dict(x=variables[1], y=variables[0], color="Convert"),
                x=cnt_tab.columns.tolist(),
                y=cnt_tab.index.tolist())
                fig.update_xaxes(side="top")
                st.plotly_chart(fig)
                # perform chi2 test
                chi2, p_value, _, _ = chi2_contingency(cnt_tab)
                st.info(f"We performed Chi-squared test on the two variables. The p-value is {round(p_value, 4)}, which means {'there is significant difference among the groups.' if p_value<0.05 else 'there is not significant difference among the groups.'}")

    with salesTab:
        # Revenue Map Component
        revenue_df, counties = get_revenue_df()
        st.metric("Revenue", f'${revenue_df.revenue.sum():,}')

        left, right = st.columns([2,4])
        with left:
            # Data prep
            policy = get_policy_df()
            agent_df = policy.assign(
                revenue=lambda x: x['quoted_amt']*x['convert_ind']  # calculate revenue per policy
            )[['Agent_cd', 'revenue']].groupby('Agent_cd', as_index=False).agg({'revenue': 'sum'})  # sum up revenue per agent
            # agent_df['Agent_cd'] = agent_df['Agent_cd'].apply(str)  # change the type of agent id into string (e.g. 32759856)
            agent_df = agent_df.sort_values('revenue', ascending=False)
            n = st.slider(
                'Top N Agent', 2, 10, 5,
                label_visibility='collapsed', help="Top N Agent")
            fig = px.bar(
                agent_df.head(n), x='Agent_cd', y='revenue',
                color_discrete_sequence=['#6E75A4']*n)
            fig.update_xaxes(type='category', tickangle=45)
            fig.update_layout(
                # xaxis=dict(autorange="reversed"),
                yaxis_title=None,
                width=350,
                bargap=0.5,
                margin={"t":0,"l":20},
                plot_bgcolor='rgba(0, 0, 0, 0)', # remove bg in plot area
                paper_bgcolor='rgba(0, 0, 0, 0)', # remove bg in figure area 
            )
            fig.update_traces(width=0.5)
            st.plotly_chart(fig)
        with right:
            granularity = st.radio(
                label="",
                options=["States", "Counties"],
                label_visibility='collapsed',
                horizontal=True,
            )
            if granularity == "States":
                states_revenue = revenue_df.groupby('state_id', as_index=False).sum()[['state_id', 'revenue']]
                fig = px.choropleth(
                    locations=states_revenue.state_id.tolist(), 
                    locationmode="USA-states", scope="usa",
                    color=states_revenue.revenue.tolist(),
                    color_continuous_scale='ice')
                fig.update_geos(fitbounds="locations", visible=True)
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig)
            if granularity == "Counties":
                fig = px.choropleth(
                    revenue_df, geojson=counties, locations='fips', 
                    color='revenue', color_continuous_scale="ice",
                    hover_data=['state_id','county_name'],
                    scope="usa")
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig)
        # More detailed analysis
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
            st.markdown("From the bar plot, we could see the number of converted customer is higher for no discounts. However, it does not represent the overall conversion rate. **We'll perform A/B test to examine the relationship between conversion and discount**.")
            fig = px.bar(
                discount_df, x=['Not converted', 'Converted'], y='discount',
                orientation ='h')
            fig.update_layout(
                xaxis_title='Freq')
            st.plotly_chart(fig)
            st.markdown("---")
            # Prepare experiment
            n_control = discount_df.sum(1)[0]
            n_test = discount_df.sum(1)[1]
            convert_control = discount_df.query('discount=="No"')['Converted'].values[0]
            convert_test = discount_df.query('discount=="Yes"')['Converted'].values[0]
            z_score, p_value = sp.proportions_ztest([convert_control, convert_test], [n_control, n_test], alternative='smaller')
            left, right = st.columns([2,4])
            with left:
                st.subheader("A/B Test")
                result = {
                    "Treatment": "Discount",
                    "Control Group Size": n_control,
                    "Treatment Group Size": n_test,
                    "Control Group Convert": convert_control,
                    "Treatment Group Convert": convert_test,
                    "p-value": round(p_value, 4)
                }
                st.dataframe(pd.DataFrame(result, index=['Result']).T)
                st.markdown(r"Our null hypothesis is $H_0: P_{No}=P_{Yes}$")
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
            # st.dataframe(predictors)
            predictorsTrans = preprocess(predictors)
            # st.dataframe(predictorsTrans)
            # make prediction
            st.metric('Conversion Rate', make_prediction(predictorsTrans, clf))
        else:
            st.info("Please submit the customer information", icon="👈")
     

if __name__ == '__main__':
    main()
import pickle
import os
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb

# st.set_option("browser.gatherUsageStats", False)
PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"wide"}
st.set_page_config(**PAGE_CONFIG)
benchmark_file = "benchmark_2.pkl"

@st.cache
def get_model(benchmark_file):
    with open(benchmark_file, 'rb') as f:
        benchmark = pickle.load(f)
    return benchmark["model"], list(benchmark["dataset"].drop(['policy_id', 'split', 'convert_ind'], 1).columns)

def preprocess(df):
    columns = df.select_dtypes(include=["object_"]).columns
    lbEncode = {'state_id': {'AL': 0, 'CT': 1, 'FL': 2, 'GA': 3, 'MN': 4, 'NJ': 5, 'NY': 6, 'WI': 7}, 'discount': {'No': 0, 'Yes': 1}, 'Prior_carrier_grp': {'Carrier_1': 0, 'Carrier_2': 1, 'Carrier_3': 2, 'Carrier_4': 3, 'Carrier_5': 4, 'Carrier_6': 5, 'Carrier_7': 6, 'Carrier_8': 7, 'Other': 8}}
    for cat_col in columns:
        df[cat_col] = df[cat_col].map(lbEncode[cat_col])
    return df

def make_prediction(df: any, model: any) -> any:
    return model.predict_proba(df)[:,1]

clf, columns = get_model(benchmark_file)
predictors = pd.DataFrame(np.zeros(len(columns)).reshape(1,-1), columns=columns)

def main():
    st.title("Streamlit App for 2022 Travelers")
    # side panel
    with st.sidebar:
        st.subheader("Inputs")
        st.markdown("**Personal Information**")
        predictors['quoted_amt'] = st.number_input(
            "Quoted Amt:",
            0.0,9999999.0,5876.0,
            step=1.0
        )
        drivers_age = st.number_input(
            "Customer age:",
            0.0,100.0,40.0,
            step=1.0
        )
        credit_score = st.number_input(
            "Customer credit score:",
            300.0,850.0,642.0,
            step=1.0
        )
        total_number_veh = st.number_input(
            "Total number vehicles on the policy:",
            1,10,3,
            step=1
        )
        Prior_carrier_grp = st.selectbox(
            "Prior carrier group:",
            ['Carrier_1','Carrier_2','Carrier_3','Carrier_4','Carrier_5','Carrier_6','Carrier_7','Carrier_8','other']
        )
        discount = st.checkbox("Discount applied")

        st.markdown("---")

        st.markdown("**Location Information**")
        state_id = st.selectbox(
        'Pick the state:',
        ['NY', 'FL', 'NJ', 'CT', 'MN', 'WI', 'AL', 'GA'])
        CAT_zone = st.select_slider(
        'CAT_zone', 
        [1,2,3,4,5])
        submitted = st.button('Submit')
    
  # main panel
    with st.expander('About this app'):
        st.markdown('This app shows the **conversion probability** based on the information you provide.')
        st.write('😊Happy Coding.')
    tab1, tab2, predictionTab = st.tabs(["Time Series", "Customer Group", "Prediction"])
    with predictionTab:
        if submitted:
            # predictors['quoted_amt'] = quoted_amt
            # predictors['total_number_veh'] = total_number_veh
            # predictors['CAT_zone'] = CAT_zone
            # predictors['state_id'] = state_id
            # predictors['discount'] = int(discount)
            # predictors['Prior_carrier_grp'] = Prior_carrier_grp
            # predictors['drivers_age'] = drivers_age
            # predictors['credit_score'] = credit_score

            st.dataframe(predictors)
            predictorsTrans = preprocess(predictors)
            st.dataframe(predictorsTrans)
            st.metric('Conversion Rate', make_prediction(predictorsTrans, clf))
        else:
            st.info("Please submit the customer information", icon="👈")
     

if __name__ == '__main__':
    main()
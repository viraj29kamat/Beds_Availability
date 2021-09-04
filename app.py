import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing


st.title('Forecasting of Beds Available')
st.sidebar.header('Select Date To Check Beds Availability ')


def user_input_features():
    d = pd.date_range(start="2021-06-16", end="2021-07-16")
    d = d.astype(str)
    select_date = st.sidebar.selectbox('Date',d)


    return select_date


df = user_input_features()

st.subheader('Date')
st.write(df)
data = pd.read_csv('https://raw.githubusercontent.com/viraj29kamat/Beds_Availability/main/Beds_Occupied.csv')
data['collection_date'] =  pd.to_datetime(data['collection_date'],infer_datetime_format=True)
data['Total_available_beds'] = 900 - data['Total Inpatient Beds']
data1 = data.drop(['Total Inpatient Beds'],axis=1)
indexedDataset = data1.set_index(['collection_date'])
missing_dates=pd.date_range(start="2020-06-15", end="2021-06-15").difference(indexedDataset.index)
r = pd.date_range(start="2020-06-15", end="2021-06-15")
newdata=indexedDataset.reindex(r).rename_axis('collection_date').reset_index()
newdata1= newdata['Total_available_beds'].interpolate(method="linear")
finaldata = newdata.copy()
finaldata['Total_available_beds'] = newdata1
#finaldata['t'] = range(1,367)
#finaldata['tsqr'] = finaldata['t']**2
#finaldata['log_Total_available_beds'] = np.log(finaldata['Total_available_beds'])
final_model = ExponentialSmoothing(finaldata["Total_available_beds"],seasonal='add',seasonal_periods=8).fit()
forecast_final = final_model.forecast(steps=31)
date = pd.date_range(start="2021-06-16", end="2021-07-16")
forecast_beds = pd.DataFrame({'date':date,'available_beds':forecast_final})
forecast_beds['date']=forecast_beds['date'].astype(str)
forecast_beds['available_beds']=forecast_beds['available_beds'].astype(int)

st.subheader('Available Beds')
for a, b in forecast_beds.itertuples(index=False):

    if a == df:
        no_of_beds = b

st.write(no_of_beds)

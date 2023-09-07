import streamlit as st
import pandas as pd
import yfinance as yf
import pygwalker as pyg
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from Pillow import Image
import plotly.express as px

############################################################################################
# SIDEBAR TITLE and MENU (menu no.-1) (automatic run becaus eit is in the main function)
# THE MAIN MENU ARE LINKED TO 

st.title('Stock Market Dashboard')

#st.image("https://www.pexels.com/photo/close-up-photo-of-monitor-159888/")
img = Image.open("pexels-leeloo-thefirst-7247399.jpg")
st.image(img)

st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by Rajib Kumar Tah")

def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict', 'Visualize by yourself'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Visualize by yourself':
        pyg.walk (data, dark= 'light')
    else:
        predict()



####################################################################################################
#FUNCTION TO DOWNLOAD DATA with YFINANCE

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

##################################################################################################
# SIDEBAR MUNU ((menu no.-2)

option = st.sidebar.selectbox('Select the stock',['RELIANCE.NS', 'ITC.NS','BEL.NS'])

option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter no. of days from today', value= 365)
before = today - datetime.timedelta(days=duration)

start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)

if st.sidebar.button('Run'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')

#####################################################################################
# CALLING THE FUNCTION download_data TO DOWNLOAD DATA

data = download_data(option, start_date, end_date)
scaler = StandardScaler()

#####################################################################################
# ADDING MORE COLUMNS TO THE data DATAFRAME AND CREATING A NEW DATAFRAME WITH THE NAME data_added_columns

data_added_columns = data
data_added_columns['SMA'] = SMAIndicator(data_added_columns.Close, window=14).sma_indicator()

####################################################################################
# MAKING AN ALL INCLUSIVE FUNCTION IN THE MAIN BODY OF THE APP WITH:
# A) DEFINING THE RADIO BUTTONS
# B) WHAT ACTION TO BE DONE IF THE RADIO BUTTION IS CLICKED
# C) THE TECHNICAL ANALYSIS CODE DRIVING THOSE ACTIONS

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['All important indicators', 'BB', 'MACD', 'RSI', 'EMA'])

    #######################################
    # CODE FOR BOLLINGER BAND
    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    ########################################

    if option == 'All important indicators':
        st.write('Close Price and SMA')
        st.line_chart(data_added_columns[['Close', 'SMA']])
        st.write('BollingerBands')
        st.line_chart(bb[['Close', 'bb_h', 'bb_l']])
        st.write('Moving Average Convergence Divergence')
        st.line_chart(MACD(data.Close).macd())
        st.write('Relative Strength Indicator')
        st.line_chart(RSIIndicator(data.Close).rsi())

        
        
    elif option == 'BB':
         st.write('BollingerBands')
         st.line_chart(bb)
        
    elif option == 'MACD':
        st.write('Close Price')
        st.line_chart(data.Close)
        st.write('Moving Average Convergence Divergence')
        st.line_chart(MACD(data.Close).macd())
        
    elif option == 'RSI':
        st.write('Close Price')
        st.line_chart(data.Close)
        st.write('Relative Strength Indicator')
        st.line_chart(RSIIndicator(data.Close).rsi())
        
    else:
        st.write('Exponential Moving Average')
        st.line_chart(EMAIndicator(data.Close).ema_indicator())

###############################################################################################

def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))



def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        else:
            engine = XGBRegressor()
            model_engine(engine, num)


def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


if __name__ == '__main__':
    main()

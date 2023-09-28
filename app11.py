import streamlit as st
import pandas as pd
import yfinance as yf
import pygwalker as pyg
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from PIL import Image
import plotly.express as px
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import time
import tensorflow as tf
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

############################################################################################
# SIDEBAR TITLE and MENU (menu no.-1) (automatic run becaus eit is in the main function)
# THE MAIN MENU ARE LINKED TO 

st.title('Stock Market Dashboard')


#st.Image("https://www.pexels.com/photo/close-up-photo-of-monitor-159888/")
img = Image.open("pexels-leeloo-thefirst-7247399.jpg")
img = img.resize((400,300))
st.sidebar.image(img)
st.sidebar.divider()

#Lottie file for streamlit animation
with st.sidebar:
    st_lottie("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")
    
st.sidebar.write(":heavy_minus_sign:" * 9) # horizontal separator line. Just change 34 as needed

st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below', icon = "🔥")
st.sidebar.info("Created and designed by Rajib Kumar Tah", icon="🤖")



def main():
    option = st.selectbox('Make a choice', ['About', 'Visualize', 'Comparison', 'Recent Data', 'Predict', 'Prediction Chart', 'Visualize by yourself', 'Contact Us'])
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    if option == 'About':
        about()
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Comparison':
        comparison()
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Visualize by yourself':
        streamlit_tableau()
    elif option == 'Predict':
        predict()
    elif option == 'Prediction Chart':
        predictionchart()
    elif option == 'Contact Us':
        contact_us()
    
     
def about():
    st.subheader("About")
    st.subheader('Developed by Rajib Kumar Tah')

    return()

####################################################################################################
#FUNCTION TO DOWNLOAD DATA with YFINANCE

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

##################################################################################################
# SIDEBAR MUNU ((menu no.-2)
stock_df = pd.read_csv("StockStreamTickersData.csv")
tickers = stock_df["Company Name"]
dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
symb_list = []  # list for storing symbols
for i in tickers:  # for each asset selected
        val = dict_csv.get(i)  # get symbol from csv file
        symb_list.append(val)  # append symbol to list

option = st.selectbox('Select the stock', symb_list) #['RELIANCE.NS', 'ITC.NS','BEL.NS']
st.write("---") #It is the same as st.divider()

option = option.upper()
today = datetime.date.today()
#duration = st.sidebar.number_input('Enter no. of days from today', value= 365) #This is a manual input system
duration = st.slider('Enter number of months to analyse:', 0,60,12) #This is a slider input system
duration = duration *30
st.write('Number of days from today:', int(duration/30),'months')
before = today - datetime.timedelta(days=duration)

#start_date = st.sidebar.date_input('Start Date', value=before) # This was to show the inputs side by side
#end_date = st.sidebar.date_input('End date', today) # This was to show the inputs side by side

col1, col2 = st.columns(2)
with col1:
   start_date = st.date_input('Start Date', value=before)
with col2:
   end_date = st.date_input('End Date', today)

if st.button('Run'):
    if start_date < end_date:
        st.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.error('Error: End date must fall after start date')

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


def comparison():
    stock_df = pd.read_csv("StockStreamTickersData.csv")
    st.subheader("Stocks Performance Comparison")
    tickers = stock_df["Company Name"]
    # dropdown for selecting assets
    dropdown = st.multiselect('Pick your assets', tickers)
    
    with st.spinner('Loading...'):  # spinner while loading
        time.sleep(2)
        # st.success('Loaded')

    dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
    symb_list = []  # list for storing symbols
    for i in dropdown:  # for each asset selected
        val = dict_csv.get(i)  # get symbol from csv file
        symb_list.append(val)  # append symbol to list
    
    def relativeret(df):  # function for calculating relative return
        rel = df.pct_change()  # calculate relative return
        cumret = (1+rel).cumprod() - 1  # calculate cumulative return
        cumret = cumret.fillna(0)  # fill NaN values with 0
        return cumret  # return cumulative return

    
    if len(dropdown) > 0:  # if user selects atleast one asset
        #Calling the above Relative return function
        df = relativeret(download_data(symb_list, start_date, end_date))['Adj Close']  # download data from yfinance
        raw_df = relativeret(download_data(symb_list, start_date, end_date))
        raw_df.reset_index(inplace=True)  # reset index
        #Closing price code
        closingPrice = download_data(symb_list, start_date, end_date)['Adj Close']  # download data from yfinance
        #Volume code
        volume = download_data(symb_list, start_date, end_date)['Volume']

        
        ### Original
        import random
        dfff = pd.DataFrame({
           "name": ["Roadmap", "Extras", "Issues"],
           "url": ["https://roadmap.streamlit.app", "https://extras.streamlit.app", "https://issues.streamlit.app"],
            "stars": [random.randint(0, 1000) for _ in range(3)],
            "views_history": [[random.randint(0, 5000) for _ in range(30)] for _ in range(3)],})
        
        st.write(dfff)
        #Calling the data which is set above 
        st.dataframe(dfff,
            column_config={
            "name": "App name",
            "url": st.column_config.LinkColumn("App URL"),
             "stars": st.column_config.NumberColumn("Github Stars", help="Number of stars on GitHub", format="%d ⭐",),
            "views_history": st.column_config.LineChartColumn("Views (past 30 days)", y_min=0, y_max=5000),}, hide_index=True,)
 










        

        
        
        dfchange= df
        dfchange = dfchange.reset_index(drop=True)
        st.write(dfchange)
        ccc = []  # list for storing symbols
        for i in dfchange.columns:  # for each asset selected
           val = dfchange.get(i)  # get symbol from csv file
           ccc.append(val)  # append symbol to list
           ccc.tolist()
        st.write(ccc)

           
        ### Revised
        aa = pd.DataFrame({
           "Stock": symb_list,
           #"chart": [[random.randint(0, 5000) for _ in range(30)] for _ in range(2)]})
           "chart": ccc })
            
        st.write(aa)

        #Calling the data which is set above 
        st.dataframe(aa,
            column_config={
            "Stock": "Stock",
            "chart": st.column_config.LineChartColumn("Chart", y_min=0, y_max=5000),}, hide_index=True,)
        
       

        #Dropdown for selecting type of chart within the above list of stocks selcted for comparison
        st.subheader('Raw Data {}'.format(dropdown))
        chart = ('Line Chart', 'Area Chart', 'Bar Chart')  # chart types
        dropdown1 = st.selectbox('Pick your chart', chart)
        with st.spinner('Loading...'):  # spinner while loading
            time.sleep(2)

        st.subheader('Comparison of  {}'.format(dropdown))
        #Type of chart dropdown conditions. Note that this if condition is indented within the above if condition suggesting that
        #this is a selection within the above selection     
        
        if (dropdown1) == 'Line Chart':  # if user selects 'Line Chart'
            # display relative comparison chart of the selected assets
            st.write("### Relative return")
            st.line_chart(df)  # display line chart
            # display closing price of selected assets
            col1, col2 = st.columns(2)
            with col1:
              st.write("### Closing Price")
              st.line_chart(closingPrice)  # display line chart
            # display volume of selected assets
            with col2:
              st.write("### Volume")
              st.line_chart(volume)  # display line chart

        elif (dropdown1) == 'Area Chart':  # if user selects 'Area Chart'
            # display relative comparison chart of the selected assets
            st.write("### Relative return")
            st.area_chart(df)  # display line chart
            # display closing price of selected assets
            col1, col2 = st.columns(2)
            with col1:
              st.write("### Closing Price")
              st.area_chart(closingPrice)  # display line chart
            # display volume of selected assets
            with col2:
              st.write("### Volume")
              st.area_chart(volume)  # display line chart
                
        elif (dropdown1) == 'Bar Chart':  # if user selects 'Bar Chart'
            # display relative comparison chart of the selected assets
            st.write("### Relative return")
            st.bar_chart(df)  # display line chart
            # display closing price of selected assets
            col1, col2 = st.columns(2)
            with col1:
              st.write("### Closing Price")
              st.bar_chart(closingPrice)  # display line chart
            # display volume of selected assets
            with col2:
              st.write("### Volume")
              st.bar_chart(volume)  # display line chart

        else:
            st.line_chart(df, width=1000, height=800, use_container_width=False)  # display line chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  # display line chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume)  # display line chart

    else:  # if user doesn't select any asset
        st.write('Please select atleast one asset')  # display message
# Stock Performance Comparison Section Ends Here


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
    df = data.tail(10)
    st.dataframe(df.style.highlight_max(axis=0))


def streamlit_tableau():
    # Adjust the width of the Streamlit page
    #st.set_page_config(page_title="Use Pygwalker In Streamlit", layout="wide")
    #st.title("Use Pygwalker In Streamlit")
    pyg_html= pyg.walk (data_added_columns, dark= 'light', return_html=True) # dark= 'light'
    components.html(pyg_html, width= 1000, height= 800, scrolling=True)
    

def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.slider('How many days forecast do you want?',0,60,5)
    #num = st.number_input('How many days forecast?', value=5)
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



def predictionchart():
    stock = option # One stock prediction chart at a time for the stock chosen in the left panel dropdown
    #stocks = ['CUBEXTUB.NS', 'AIAENG.NS',  'ASTEC.NS'] # if bultiple stock predictions are required
    pd.options.mode.chained_assignment = None
    tf.random.set_seed(0)

    stock_short = yf.Ticker(stock)
    df = stock_short.history(start= date.today()-timedelta(120), end= date.today(), interval= '1d')
    y = df['Close'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 70  # length of input sequences (lookback period)
    n_forecast = 5  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs= 30, batch_size=32, verbose=0)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = df[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace= True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan
    results = pd.concat([df_past, df_future]).set_index('Date')
   
    #Visualisation 
    st.write('Prediction of: ', stock)
    fig = px.line(results, title="Chart")
    fig.update_layout(height=500, width=700, font_family="Courier New", font_color="blue", title_font_family="Times New Roman", title_font_color="red", legend_title_font_color="green", title_font_size=40)
    fig.update_xaxes(showgrid=False, rangeslider_visible=False, rangeselector=dict(buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all"),])),)
    fig.update_yaxes(showgrid=True)
    #fig.update_layout(height=300, width=600, font_family="Courier New", font_color="blue", title_font_family="Times New Roman", title_font_color="red", legend_title_font_color="green", title_font_size=40))
    fig.update_layout(
    font_family="Courier New",
    font_color="green",
    title_font_family="Times New Roman",
    title_font_color="Purple",
    legend_title_font_color="green",
    title_font_size=15,)
    st.plotly_chart(fig, theme="streamlit")
    #st.plotly_chart(fig, theme=None)

           
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

def contact_us():
    #Form submit template
    st.header(':mailbox: Get in touch with me!')    
    contact_form= """
    <form action="https://formsubmit.co/rajibtah@gmail.com" method="POST"/>
         <input type="text" name="name" required>
         <input type="email" name="email" required>
         <button type="submit">Send</button>
    </form>
    """
    st.contact_form = st.markdown(contact_form, unsafe_allow_html= True) 



if __name__ == '__main__':
    main()

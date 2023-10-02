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
from io import BytesIO
from IPython.display import display, HTML
import base64
from streamlit_option_menu import option_menu
from streamlit_card import card

#Page configuration settings. Note the icon on the left top corner and the menus on the dot dot dot on the right top corner
#This settings are to be put before writing any other code in the page just after the library imports are complete
st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug", # Mention the url this option should point to
        'About': "# Here comes this *extremely* cool stock market app!"}) # Mention the url this option should point to


with st.sidebar:
    selected = option_menu("Utilities", ["FirstMenu", "SecondMenu", "ThirdMenu"])

    if(selected == 'FirstMenu'):
        tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
        with tab1:
           st.header("A cat")
           st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
        with tab2:
           st.header("A dog")
           img = Image.open("pexels-leeloo-thefirst-7247399.jpg") 
           #transposing image 
           transposed_img = img.transpose(Image.FLIP_LEFT_RIGHT)
          
           
          
           st.image(transposed_img, width=200)
            
    elif(selected == 'SecondMenu'):
        st.subheader("Second Menu Job")
    elif(selected == 'ThirdMenu'):
        st.subheader("Third Menu job")
    
############################################################################################

st.title('Stock Market Dashboard')


#st.Image("https://www.pexels.com/photo/close-up-photo-of-monitor-159888/")
img = Image.open("pexels-leeloo-thefirst-7247399.jpg")
img = img.resize((400,300))
with st.expander("See explanation"):
    st.write('Hello, *World!* :sunglasses:')
    
st.sidebar.image(img)

st.sidebar.divider()

#card form, This crad will point to a website
hasClicked = card(
  title="Hello World!",
  text="Some description",
  image="http://placekitten.com/200/300",
  url="https://sail.co.in"
)

#This card if clicked will do something
hasClicked = card(
  title="Hello World!",
  text="Some description",
  image="http://placekitten.com/20/30",
  on_click=lambda: print("Do ths job!"))

from streamlit_card import card

res = card(
    title="Streamlit Card",
    text="This is a test card",
    image="https://placekitten.com/500/500",
    styles={
        "card": {
            "width": "500px",
            "height": "500px",
            "border-radius": "60px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)"},
        "text": {
            "font-family": "serif",}})


col1, col2, col3 = st.columns((2,1,1))
original = Image.open("pexels-leeloo-thefirst-7247399.jpg")
col1.header("Original")
col1.image(original, use_column_width=True)

col2.header("Trans")
transposed_img = img.transpose(Image.FLIP_LEFT_RIGHT)
col2.image(transposed_img, use_column_width=True)

original = original
width, height = original.size
area = (0, 0, width/2, height/2)
original = original.crop(area)
#Saved in the same relative location
original.save("cropped_picture.jpg") 
grayscale = original.convert('LA')
#grayscale = grayscale.rotate(60) #this is required if we want to rotate the picture
col3.header("Grayscale")
col3.image(grayscale, use_column_width=True)

#To get a grid layout with st.container
st.title("Let's create a table!")
for i in range(1, 10):
    cols = st.columns(4)
    cols[0].write(f'{i}')
    cols[1].write(f'{i * i}')
    cols[2].write(f'{i * i * i}')
    cols[3].write('x' * i)

#To get a grid layout with st.container
st.title("Let's create a table!")
for i in range(1, 10):
    cols = st.columns(4)
    cols[0].write('ABC')
    cols[1].write(f'{i * i}')
    cols[2].write(f'{i * i * i}')
    cols[3].write('x' * i)




from itertools import cycle

filteredImages = ["https://static.streamlit.io/examples/cat.jpg" "https://static.streamlit.io/examples/cat.jpg",
"https://static.streamlit.io/examples/cat.jpg", "https://static.streamlit.io/examples/cat.jpg",
"https://static.streamlit.io/examples/cat.jpg", "https://static.streamlit.io/examples/cat.jpg"] # your images here
caption = [] # your caption here
cols = cycle(st.columns(4)) # st.columns here since it is out of beta at the time I'm writing this
for idx, filteredImage in enumerate(filteredImages):
    next(cols).image(filteredImage, width=150, caption=caption[idx])

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

        #sparklines #https://github.com/iiSeymour/sparkline-nb/blob/master/sparkline-nb.ipynb
        dfstack= closingPrice.stack()
        #dfstack = dfstack.to_frame()
        dfstack.index.set_names('Stock', level=len(dfstack.index.names)-1, inplace=True)
        dfstack = dfstack.reset_index().rename(columns={0:'Value'})
        rates = dfstack.groupby(['Stock']).aggregate({'Value': sparkline})
        st.write(HTML(rates.to_html(escape=False)))
        
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

def sparkline(data, figsize=(4, 0.25), **kwags):
    """
    Returns a HTML image tag containing a base64 encoded sparkline style plot
    """
    data = list(data)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwags)
    ax.plot(data)
    for k,v in ax.spines.items():
        v.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])    

    plt.plot(len(data) - 1, data[len(data) - 1], 'r.')

    ax.fill_between(range(len(data)), data, len(data)*[min(data)], alpha=0.1)
    
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    plt.close()
    return '<img src="data:image/png;base64,{}"/>'.format(base64.b64encode(img.read()).decode())


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

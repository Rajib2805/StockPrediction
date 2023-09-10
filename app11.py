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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from PIL import Image
import plotly.express as px
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie

import webbrowser
import sqlite3
from user import User

############################################################################################
# SIDEBAR TITLE and MENU (menu no.-1) (automatic run becaus eit is in the main function)
# THE MAIN MENU ARE LINKED TO 

st.title('Stock Market Dashboard')


#st.Image("https://www.pexels.com/photo/close-up-photo-of-monitor-159888/")
img = Image.open("pexels-leeloo-thefirst-7247399.jpg")
st.image(img)

#Lottie file for streamlit animation
with st.echo():
    st_lottie("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")
    
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by Rajib Kumar Tah")

def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict', 'Visualize by yourself'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Visualize by yourself':
        streamlit_tableau()
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
stock_df = pd.read_csv("StockStreamTickersData.csv")
tickers = stock_df["Company Name"]
dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
symb_list = []  # list for storing symbols
for i in tickers:  # for each asset selected
        val = dict_csv.get(i)  # get symbol from csv file
        symb_list.append(val)  # append symbol to list

option = st.sidebar.selectbox('Select the stock', symb_list) #['RELIANCE.NS', 'ITC.NS','BEL.NS']

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


def streamlit_tableau():
    # Adjust the width of the Streamlit page
    #st.set_page_config(page_title="Use Pygwalker In Streamlit", layout="wide")
    #st.title("Use Pygwalker In Streamlit")
    pyg_html= pyg.walk (data_added_columns, dark= 'light', return_html=True) # dark= 'light'
    components.html(pyg_html, width= 1000, height= 800, scrolling=True)
    

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

#Form submit template
st.header(':mailbox: Get in touch with me!')    
contact_form= '''
<form action="https://formsubmit.co/rajibtah@gmail.com" method="POST"/>
     <input type="text" name="name" required>
     <input type="email" name="email" required>
     <button type="submit">Send</button>
</form>
'''
st.contact_form = st.markdown(contact_form, unsafe_allow_html= True) 







import streamlit as st 
import webbrowser
import sqlite3
from user import User

st.set_page_config(page_title="PMGR", page_icon="🚀" )     
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("Image-of-your-choice");
background-size: 100%;
display: flex;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


conn = sqlite3.connect("pwd.db")
c = conn.cursor()
c.execute("""CREATE TABLE if not exists pwd_mgr (app_name varchar(20) not null,
                        user_name varchar(50) not null,
                        pass_word varchar(50) not null,
                        email_address varchar(100) not null,
                        url varchar(255) not null,
                    primary key(app_name)       
                    );""")


def insert_data(u):
    with conn:
        c.execute("insert into pwd_mgr values (:app, :user, :pass, :email, :url)", 
                  {'app': u.app, 'user': u.username, 'pass': u.password, 'email': u.email, 'url': u.url})
        
def get_cred_by_app(app):
    with conn:
        c.execute("select app_name, user_name, pass_word, email_address, 
                   url FROM pwd_mgr where app_name = :name;", {'name': app})
        return c.fetchone()
    
def remove_app_cred(app):
    with conn:
        c.execute("DELETE from pwd_mgr WHERE app_name = :name", {'name': app})
        
def update_password(app,new_pass_word):
    with conn:
        c.execute("update pwd_mgr set pass_word = :pass where app_name = :name", 
                  {'name': app, 'pass': new_pass_word})


st.title("Password Manager 🔐")
st.markdown('#')

c.execute("select count(*) from pwd_mgr")
db_size = c.fetchone()[0] 

c.execute("select app_name from pwd_mgr")
app_names = c.fetchall()
app_names = [i[0] for i in app_names]

radio_option = st.sidebar.radio("Menu", options=["Home", "Add Account", "Update Password", "Delete Account"])

if radio_option=="Home":    
    st.subheader("Find Credential 🔎")  
    st.markdown("#####")  
    if db_size>0:                   
        option = st.selectbox('Select Application 📱', app_names) # TO be populated from DB
        st.markdown("#####")        
        cred = get_cred_by_app(option)
        with st.container():    
            st.text(f"Username 👤")
            st.code(f"{cred[1]}", language="python")
            st.text_input('Password 🔑', value=cred[2], type="password",)    
            st.markdown("####")
            url = cred[4]   
            if st.button('Launch 🚀', use_container_width=True):
                webbrowser.open_new_tab(url=url)                
        st.markdown('##')    
        with st.expander("Additional Details:"):
            st.text(f"email")
            st.code(f"{cred[3]}", language="python")
            st.text(f"URL")
            st.code(f"{cred[4]}", language="python")
    else:
        st.info('Database is Empty.', icon="ℹ️")

if radio_option=="Add Account": 
    st.subheader("Add New Credential 🗝️")
    st.markdown("####")    
    app_name = st.text_input('Application 📱', 'Twitter')
    user_name = st.text_input('User Name 👤', 'tweety')
    pass_word = st.text_input('Password 🔑', 'pass123', type="password",)
    email = st.text_input('Email 📧', 'tweety@xyz.com')
    url = st.text_input('Website 🔗', 'twitter.com')
    st.markdown("####")
    if st.button('Save ⏬', use_container_width=True):
        try:
            data = User(app_name, user_name, pass_word, email, url)
            insert_data(data)
            st.success(f"{app_name}'s credential is added to the Database!", icon="✅")
        except:
            st.warning('Something went wrong! Try Again.', icon="⚠️")
    st.markdown("####")
    st.info(f"Available Credentials in Database: {db_size}", icon="💾") 
    
if radio_option=="Update Password": 
    st.subheader("Update Password 🔄")
    st.markdown('#####')   
    if db_size>0: 
        up_app = st.selectbox('Select an Account you want to update 👇', app_names) 
        st.markdown('####')
        new_pass_1 = st.text_input('New Password ', 'new123', type="password",)
        new_pass_2 = st.text_input('Confirm New Password', 'new123', type="password",)
        if new_pass_1==new_pass_2:
                          
            if st.button('Update ⚡️', use_container_width=True):
                try:
                    update_password(up_app,new_pass_1)
                    st.success(f"{up_app}'s password is updated!", icon="✅")
                except:
                    st.info(' Database is Empty. Go to Create to add Data ⬅️', icon="ℹ️")    
        else:
            st.warning("Password don't match! Try Again.", icon="⚠️")
    else:
        st.info('Database is Empty.', icon="ℹ️")
   
if radio_option=="Delete Account":
    st.subheader("Delete Credential 🗑️")  
    st.markdown("#####")     
    if db_size>0: 
        agree = st.checkbox('View Full Database')
        if agree:
            c.execute("select app_name, email_address, url from pwd_mgr")
            results = c.fetchall()
            st.table(results)        
        st.markdown('#####')      
        delt = st.selectbox('Select an Account you want to delete 👇', app_names) 
        st.markdown('####')              
        if st.button('Delete ❌', use_container_width=True):
            try:
                remove_app_cred(delt)
                st.success(f"{delt}'s Credential is removed from Database!", icon="✅")
            except:
                st.info(' Database is Empty. Go to Create to add Data ⬅️', icon="ℹ️")             
    else:
        st.info('Database is Empty.', icon="ℹ️")




if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta, datetime
import time
from sqlalchemy import create_engine, Integer, Float, BIGINT
import numpy as np

def go_sleep(i:int, times_i: int = 5, sec_i : int = 15):
    if i%times_i == 0:
        time.sleep(sec_i)

def twse_data_organization(df, seq):
    df = df.apply(lambda x: x.str.replace(r"[,/+X]", "", regex= True) if x.dtypes == "object" else x)
    df = df.fillna(0)
    # 資料類型轉換
    data_types = {
        "stock" : {'date' : int, 'Trading_volume' : int, 'Trading_value' : int,
                    'Number_of_trades' : int,
                    'Opening_price' : float, 'Highest_price' : float, 'Lowest_price' : float,
                    'Closing_price' : float, 'Price_difference' : float
                    },
        "market" : {'date' : int,
                    "market_trading_value": int, "weighted_Index": float,
                    "market_point_change": float}, 
        "dividend" : {'date' : int,
                    "Dividend_Yield(%)": float, "Price-to-Earnings_Ratio": float, 
                    "Price_to_Book_Ratio": float}, 
        "margin_borrow" : {'date' : int,
                    "quantity_margin": int, "amount_margin" : int,
                    "quantity_borrowed" : int, "amount_borrowed" : int}, 
        "foreign" : {'date' : int, 
                    "Foreign_Investor_Buy" : int, "Foreign_Investor_Sell" : int, 
                    "Foreign_Investor_Net_Buy" : int,
                    "Foreign_Proprietary_Buy" : int, "Foreign_Proprietary_Sell" : int,
                    "Foreign_Proprietary_Net_Buy" : int, 
                    "Investment_Buy" : int, "Investment_Sell" : int, "Investment_Net_Buy" : int,
                    "Dealer_Net_Buy" : int,
                    "Dealer_Buy(proprietary)" : int, "Dealer_Sell(proprietary)" : int, 
                    "Dealer_Net Buy(proprietary)" : int,
                    "Dealer_Buy(hedging)" : int, "Dealer_Sell(hedging)" : int, 
                    "Dealer_Net Buy(hedging)" : int,
                    "Net_Buys" : int}
    }
    df = df.astype(data_types[seq])
    return df

def download_stock(stockNo : int = 2330, delta_month: int = 10, sleep_time: int = 15, start_date:int = None, end_date: int = None):
    # Set date 
    today = date.today()
    if (start_date and end_date) != None:
        start_date_e = datetime.strptime(str(start_date), "%Y%m%d").replace(day=1)
        end_date_e = datetime.strptime(str(end_date), "%Y%m%d").replace(day=1)
        current_date = start_date_e
        dates = []
        while (current_date.year < end_date_e.year) or ((current_date.year == end_date_e.year) and current_date.month < end_date_e.month):
            dates.append(current_date)
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
            
        dates = [i.strftime("%Y%m%d") for i in dates]
        print("model A")
    else:
        dates = [(today - timedelta(days=30 * i)).strftime("%Y%m%d") for i in range(delta_month)]
        print("model B")
    print(dates)

    # set download url
    twse_url = 'https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date='
    market_url = "https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date="
    dividend_yield_url = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU?date="
    margin_borrow_url = "https://www.twse.com.tw/rwd/zh/afterTrading/TWTASU?date="
    foreign_url = "https://www.twse.com.tw/rwd/zh/fund/T86?date="
    urls = [twse_url, market_url, dividend_yield_url, margin_borrow_url, foreign_url]

    #=====================
    # stock contain list
    seq_list = ["stock", "market", "dividend", "margin_borrow", "foreign"]
    seq_list1 = ["stock", "market", "dividend"]
    
    # Column names
    stock_column = ['date', 'Trading_volume', 'Trading_value', 'Opening_price', 'Highest_price', 'Lowest_price',
            'Closing_price', 'Price_difference', 'Number_of_trades']
    
    market_column = ["date", "成交股數", "market_trading_value", "成交筆數", "weighted_Index", "market_point_change"]
    
    dividend_yield_column = ["date", "Dividend_Yield(%)", "Dividend_Year", 
                              "Price-to-Earnings_Ratio", "Price_to_Book_Ratio",
                              "Financial_Report_Year/Quarter"]
    
    dividend_yield_column_bf201703 = ["date", "Price-to-Earnings_Ratio", "Dividend_Yield(%)", "Price_to_Book_Ratio",
                                      ] 
    #["Dividend_Year", "Financial_Report_Year/Quarter"]
    
    margin_borrow_column = ["stock", "quantity_margin", "amount_margin", "quantity_borrowed", "amount_borrowed"]
    
    foreign_column = ["stockNo", "stock_name", 
                      "Foreign_Investor_Buy", "Foreign_Investor_Sell", "Foreign_Investor_Net_Buy",
                      "Foreign_Proprietary_Buy", "Foreign_Proprietary_Sell",
                      "Foreign_Proprietary_Net_Buy", 
                      "Investment_Buy", "Investment_Sell", "Investment_Net_Buy",
                      "Dealer_Net_Buy", 
                      "Dealer_Buy(proprietary)", "Dealer_Sell(proprietary)", "Dealer_Net Buy(proprietary)",
                      "Dealer_Buy(hedging)", "Dealer_Sell(hedging)", "Dealer_Net Buy(hedging)",
                      "Net_Buys"]
    
    foreign_column_s = ["stockNo", "stock_name",
                      "Foreign_Investor_Buy", "Foreign_Investor_Sell", "Foreign_Investor_Net_Buy", 
                      "Investment_Buy", "Investment_Sell", "Investment_Net_Buy",
                      "Dealer_Net_Buy", 
                      "Dealer_Buy(proprietary)", "Dealer_Sell(proprietary)", "Dealer_Net Buy(proprietary)",
                      "Dealer_Buy(hedging)", "Dealer_Sell(hedging)", "Dealer_Net Buy(hedging)",
                      "Net_Buys"]
    
    columns = [stock_column, market_column, dividend_yield_column, margin_borrow_column, foreign_column]
    
    # sleep
    slp = 0
    
    # create stock dataframe
    df1 = pd.DataFrame()
    for seq, url, column in zip(seq_list, urls, columns):
        dfs = []
        # focus on common font
        if seq in seq_list1:
            for n, i in enumerate(dates):
            ## download from twse
                print(seq, n, i)
                if seq == "market":
                    url1 = url + str(i)
                else:
                    url1 = url + str(i) + "&stockNo=" + str(stockNo)
                response = requests.get(url1).json()
                data = response["data"]
                if (seq == "dividend") and ((int(i)//100) <= 201703):
                    mid_df = pd.DataFrame(data= data, columns= dividend_yield_column_bf201703)
                else:
                    mid_df = pd.DataFrame(data= data, columns= column)
                dfs.append(mid_df)
                print(f"{seq} - {n+1}/{len(dates)} - {int((n + 1) / len(dates) * 100)} %")
                slp += 1
                go_sleep(slp)
            ## data organization
            if seq == "stock":
                df = pd.concat(dfs, ignore_index= True)
                df = twse_data_organization(df,seq)
                df["date"] = df["date"].astype(int) + 19110000
            elif seq in ["market", "dividend"]:
                df1 = pd.concat(dfs, ignore_index= True)
                if seq == "market":
                    df1 = df1.drop(["成交股數", "成交筆數"], axis= 1)
                    df1 = twse_data_organization(df1,seq)
                elif seq == "dividend":
                    if (int(i)//100) > 201703:
                        df1 = df1.drop(["Dividend_Year", "Financial_Report_Year/Quarter"], axis= 1)
                    df1["date"] = df1["date"].str.replace(r"[年, 月, 日]", "", regex= True)
                    df1 = twse_data_organization(df1,seq)
                df1["date"] = df1["date"] + 19110000
                df = pd.merge(df, df1, on= "date")
            
        # focus on margin_borrow and foreign
        elif seq not in seq_list1:
            day_list = [i for i in df["date"].tolist() if i != int(today.strftime("%Y%m%d"))]
            ## download from twse
            foreign_change_date = 20171218
            for n, day in enumerate(day_list):
                if seq == "margin_borrow":
                    url1 = url + str(day)
                elif seq == "foreign":
                    url1 = url + str(day) + "&selectType=24"
                else:
                    print("which sequence?")
                response = requests.get(url1).json()
                slp += 1
                go_sleep(slp)
                ## data organization
                data = response["data"]
                if (seq == "foreign") and (int(day) < foreign_change_date):
                    mid_df = pd.DataFrame(data= data, columns= foreign_column_s)
                    for i in ["Foreign_Proprietary_Buy", "Foreign_Proprietary_Sell", "Foreign_Proprietary_Net_Buy"]:
                        mid_df[i] = 0
                else:
                    mid_df = pd.DataFrame(data= data, columns= column)
                print(f"{seq} - {n+1}/{len(day_list)} - {int((n + 1) / len(day_list) * 100)} %\n{url1}")
                if seq == "margin_borrow":
                    mid_df1 = mid_df.loc[mid_df["stock"].str.contains(str(stockNo))]
                    mid_df = mid_df1.iloc[:, 1:]
                elif seq == "foreign":
                    mid_df1 = mid_df.loc[mid_df["stockNo"].str.contains(str(stockNo))]
                    mid_df = mid_df1.iloc[:, 2:]
                else:
                    print("mid_df problem")
                mid_df = mid_df.copy()
                mid_df.loc[:, "date"] = day
                dfs.append(mid_df)
            df1 = pd.concat(dfs, ignore_index= True)
            df1.to_csv("temp")
            df1 = df1.fillna(0)
            df1 = twse_data_organization(df1,seq)
            df = pd.merge(df, df1, on= "date")
    df = df.sort_values("date", ignore_index= True)
    return df

def add_future_result(df, delta_days: int = 5):
    conv = {"weighted_Index" : "market_feature",
            "Closing_price" : "stock_feature"}
    #delta_days = delta_days - 1
    for dif_targ in ["weighted_Index","Closing_price"]:
        profit_loss_result = []
        for i in range(df.shape[0]- delta_days):
            diff = df.at[i + delta_days, dif_targ] - df.at[i, dif_targ]
            pct_diff = diff/df.at[i, dif_targ]
            # classified promising, soso, and loss
            if pct_diff >= 0.03:
                feature = 2  # Promising
            elif -0.03 <= pct_diff <= 0.03:
                feature= 1  # So-so
            else:
                feature = 0  # Loss        
            profit_loss_result.append(feature)
        profit_loss_result = profit_loss_result + [3 for _ in range(delta_days)] # fill df.index # 3= unknow
        column_name = conv[dif_targ]
        df[str(column_name)] = pd.Series(profit_loss_result)
    return df

def save_to_mysql_pd(df, un="root", password=None, host= "localhost", port = "3306", database = "python_database", table = "stock_2330"):
    if password is None:
        print("input your password")
    else:
        try:
            engine = create_engine(f'mysql+mysqlconnector://{un}:{password}@{host}:{port}/{database}')
            df.to_sql(name= table, con= engine, if_exists='replace', index=False, dtype={
                    'date':  Integer,
                    'Trading_volume': Integer,
                    'Trading_value': BIGINT,
                    'Opening_price': Float,
                    'Highest_price': Float,
                    'Lowest_price': Float,
                    'Closing_price': Float,
                    'Price_difference': Float,
                    'Number_of_trades': Integer,
                    'market_trading_value': BIGINT,
                    'weighted_Index': Float,
                    'market_point_change': Float,
                    "market_feature": Integer,
                    "stock_feature": Integer,
                    "quantity_margin": Integer,
                    "amount_margin": BIGINT, 
                    "quantity_borrowed": Integer, 
                    "amount_borrowed": BIGINT,
                    "Dividend_Yield(%)": Float, 
                    "Price-to-Earnings_Ratio": Float, 
                    "Price_to_Book_Ratio": Float,
                    "amount_borrowed": BIGINT,
                    "Foreign_Investor_Buy": BIGINT,
                    'Foreign_Investor_Sell': BIGINT, 
                    'Foreign_Investor_Net_Buy': BIGINT,
                    'Foreign_Proprietary_Buy': Integer, 
                    'Foreign_Proprietary_Sell': Integer,
                    'Foreign_Proprietary_Net_Buy': Integer,
                    'Investment_Buy': BIGINT, 
                    'Investment_Sell': BIGINT,
                    'Investment_Net_Buy': BIGINT, 
                    'Dealer_Net_Buy': BIGINT,
                    'Dealer_Buy(proprietary)': BIGINT,
                    'Dealer_Sell(proprietary)': BIGINT, 
                    'Dealer_Net Buy(proprietary)': BIGINT,
                    'Dealer_Buy(hedging)': BIGINT, 
                    'Dealer_Sell(hedging)': BIGINT,
                    'Dealer_Net Buy(hedging)': BIGINT,
                    'Net_Buys': BIGINT
            })
            print("successed save to mysql")
        except Exception as e:
            print(e)

def read_mysql(un= "root", password= None, host= "localhost", port= "3306", database= "python_database", table= "stock_2330"):
    if password is None:
        print("please input password!")
    else:
        try:
            engine = create_engine(f'mysql+mysqlconnector://{un}:{password}@{host}:{port}/{database}')
            df = pd.read_sql(f"select * from {table}", con= engine)
        except Exception as e:
            print(e)
    return df

def older_data_to_extMySQL(start_date = None, end_date= None,un= "root", password= None, host= "localhost", port= "3306", database= "python_database", table= "stock_2330"):
    if (start_date == None):
        print("date error!")
        return
    df_mysql = read_mysql(un= un, password= password, host= host, port= port, database= database, table= table)
    df_mysql.to_csv("backup")
    dif =df_mysql.iat[0, 0]//100 - int(start_date)//100
    if  dif <= 0:
        print("start date error")
        return
    df_download = download_stock(start_date= start_date, end_date= df_mysql.iat[0, 0])
    df_download.to_csv("temp")
    df = pd.concat([df_mysql, df_download], ignore_index= True)
    df = df.drop_duplicates(subset= ["date"])
    df = df.sort_values("date", ignore_index=True)
    df= df.drop(["market_feature", "stock_feature"], axis= 1)
    df = add_future_result(df)
    save_to_mysql_pd(df, password= "kuenhencheng")
    print("successed: older_data_to_extMySQL")
    print(df.shape)
    return df
  
psw = input("enter mysql password")

#older_data_to_extMySQL(20230601, password=psw)

download_stock
















###########################################################################################################
def data_to_LSTM_font(psw):
    df = read_mysql(password= psw)
    selected_features = ['Trading_volume', 'Trading_value', 'Opening_price', 'Highest_price', 'Lowest_price', 'Closing_price', 'Price_difference', 'Number_of_trades']

    # preprocessing of date data
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d") # Convert the date column to timestamp format
    df.set_index("date", inplace= True) # set date to index
    
    # serilization of data
    data = []
    target = []
    
    # 以多久天前的資料，預測幾天後的資料    
    past_days = 30
    future_days = 5
    for i in range(past_days, len(df)):
        data.append(df.values[i-past_days : i])
        target.append(df["stock_feature"].values[i])
    
    # convert the data to a 3D array
    data = np.array(data)
    target = np.array(target)
    print(data.shape, target.shape)


#data_to_LSTM_font(psw)































# download and save to csv!!!

#df = download_stock_csv(delta_month=1)
#===================================================================

# read mysql and backup!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# df_mysql = pd.read_csv("backup").iloc[:, 1:]
# save_to_mysql_pd(df_mysql, password="kuenhencheng")
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# read csv
#==================================
# df_csv = pd.read_csv("temp")
# df_csv = df_csv.iloc[:, 1:]
#==================================

# concat
#==================================
#df = pd.concat([df_mysql, df_csv], ignore_index= True)
#df = df.drop_duplicates(subset= ["date"])
#==================================


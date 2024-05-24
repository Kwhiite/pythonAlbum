import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Activation, LSTM, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras import regularizers
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder

psw = input("enter your password:\n")

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

def data_to_LSTM_font(psw):
    df = read_mysql(password= psw)
   
    # preprocessing of date data
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d") # Convert the date column to timestamp format
    df.set_index("date", inplace= True) # set date to index
    df = df.drop(columns=['Price_difference', 'Number_of_trades', 'market_point_change',  'amount_margin', 'amount_borrowed', 'Foreign_Investor_Buy', 'Foreign_Investor_Sell', 'Foreign_Proprietary_Buy', 'Foreign_Proprietary_Sell', 'Foreign_Proprietary_Net_Buy', 'Investment_Buy', 'Investment_Sell', 'Dealer_Buy(proprietary)', 'Dealer_Sell(proprietary)', 'Dealer_Net Buy(proprietary)', 'Dealer_Buy(hedging)', 'Dealer_Sell(hedging)', 'Dealer_Net Buy(hedging)', 'Dealer_Buy(proprietary)', 'Dealer_Sell(proprietary)', 'Dealer_Net Buy(proprietary)', 'Dealer_Buy(hedging)', 'Dealer_Sell(hedging)', 'Dealer_Net Buy(hedging)',  'market_feature'])
    
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # statistics hell
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    tmp = pd.DataFrame()
    # 計算移動平均線（Moving Average）
    df['MA'] = df['Closing_price'].rolling(window=5).mean()

    # 計算相對強弱指標（Relative Strength Index，RSI）
    window = 14
    delta = df['Closing_price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=window).mean()
    average_loss = loss.rolling(window=window).mean()
    rs = average_gain / average_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 計算隨機震盪指標（Stochastic Oscillator）
    n = 14
    lowest_low = df['Lowest_price'].rolling(window=n).min()
    highest_high = df['Highest_price'].rolling(window=n).max()
    df['%K'] = 100 * (df['Closing_price'] - lowest_low) / (highest_high - lowest_low)
    df['%D'] = df['%K'].rolling(window=3).mean()

    # 計算威廉指標（Williams %R）
    df['%R'] = -100 * (highest_high - df['Closing_price']) / (highest_high - lowest_low)

    # 計算MACD指標（Moving Average Convergence Divergence）
    short_ema = df['Closing_price'].ewm(span=12, adjust=False).mean()
    long_ema = df['Closing_price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    
    # 計算布林通道指標（Bollinger Bands）
    std = df['Closing_price'].rolling(window=20).std()
    df['UpperBand'] = df['MA'] + 2 * std
    df['LowerBand'] = df['MA'] - 2 * std

    # 計算成交量指標（Volume）
    df['Volume MA'] = df['Trading_volume'].rolling(window=5).mean()
    
    # 計算黃金交叉和死亡交叉（Golden Cross and Death Cross）
    short_ma = df['Closing_price'].rolling(window=50).mean()
    long_ma = df['Closing_price'].rolling(window=200).mean()
    df['GoldenCross'] = np.where(short_ma > long_ma, 1, 0)
    df['DeathCross'] = np.where(short_ma < long_ma, 1, 0)
    
    # 順勢指標（Commodity Channel Index，CCI）
    tmp.drop(tmp.index, inplace=True)
    tmp['Typical Price'] = (df['Highest_price'] + df['Lowest_price'] + df['Closing_price']) / 3
    moving_average_period = 20
    tmp['Moving Average'] = tmp['Typical Price'].rolling(moving_average_period).mean()
    tmp['Mean Deviation'] = np.abs(tmp['Typical Price'] - tmp['Moving Average'])
    cci_constant = 0.015
    df['CCI'] = (tmp['Typical Price'] - tmp['Moving Average']) / (cci_constant * tmp['Mean Deviation'])
    

    df.to_csv("tmp")
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    ############################################################################################################
    select_column = ['Trading_volume', 'Trading_value', 'Opening_price', 'Highest_price',
       'Lowest_price', 'Closing_price', 'market_trading_value',
       'weighted_Index', 'Dividend_Yield(%)', 'Price-to-Earnings_Ratio',
       'Price_to_Book_Ratio', 'quantity_margin', 'quantity_borrowed',
       'Foreign_Investor_Net_Buy', 'Investment_Net_Buy', 'Dealer_Net_Buy',
       'Net_Buys', 'stock_feature', 'MA', 'RSI', '%K', '%D', '%R', 'MACD',
       'UpperBand', 'LowerBand', 'Volume MA', 'GoldenCross', 'DeathCross',
       'CCI']
    
    select_column1 = ['Trading_volume', 'Closing_price', 'market_trading_value',
       'weighted_Index', 'Dividend_Yield(%)', 'Price-to-Earnings_Ratio',
       'Price_to_Book_Ratio', 'quantity_margin', 'quantity_borrowed',
       'Foreign_Investor_Net_Buy', 'Investment_Net_Buy', 'Dealer_Net_Buy',
       'Net_Buys', 'stock_feature', 'MA', 'RSI', '%K', '%D', '%R', 'MACD',
       'UpperBand', 'LowerBand', 'Volume MA', 'GoldenCross', 'DeathCross',
       'CCI']
    ############################################################################################################
    # serilization of data
    data = []
    target = []
    
    # 以多久天前的資料，預測幾天後的資料    
    past_days = 20
    subsequence_length = 5
    future_days = 5
    print(df.shape)
    for i in range(past_days, len(df)):
        sequence = df.iloc[i - past_days : i]  # 获取每个30天时间跨度的序列

        num_subsequences = past_days // subsequence_length  # 每个30天时间跨度切割成多少个5天时间跨度的子序列
        for j in range(num_subsequences):
            start_index = j * subsequence_length
            end_index = start_index + subsequence_length
            subsequence = sequence.iloc[start_index:end_index, :-1].values
            label = sequence.iloc[end_index - 1, -1]
            data.append(subsequence)
            target.append(label)
            # data.append(subsequence[select_column])
            # target.append(subsequence["stock_feature"])
            
        # for i in range(past_days+20, len(df)-future_days):
        #     data.append(df[select_column].values[i-past_days : i])
        #     target.append(df["stock_feature"].values[i])
    
    # convert the data to a 3D array
    data = np.array(data)
    target = np.array(target)
    data = data.reshape(-1, past_days, subsequence_length, df.shape[1] - 1)  # 改變形狀為 (1631, 20, 5, 29)
    target = target.reshape(-1,)  # 改變形狀為 (1631,)
    print(data.shape, target.shape)
    return data, target

def lstm_model(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size= 0.2, shuffle= False)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    feature = data.shape[2]

    model = Sequential([
        LSTM(feature*2, activation= "sigmoid", recurrent_activation= "sigmoid", input_shape=(20, feature), return_sequences= True),
        LSTM(feature, activation= "sigmoid", recurrent_activation= "sigmoid", return_sequences= True),
        LSTM(int(feature), activation= "sigmoid", recurrent_activation= "sigmoid", return_sequences= True),
        LSTM(int(feature*0.5), activation= "sigmoid", recurrent_activation= "sigmoid", return_sequences= True),
        LSTM(int(feature*0.5), activation= "sigmoid", recurrent_activation= "sigmoid", return_sequences= True),
        LSTM(6, activation= "relu",),
        Dense(5, activation= "relu"),
        Dense(3, activation= "softmax"),
    ])

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.fit(x_test, y_test, batch_size= 10, epochs = 30, verbose= 1, validation_split = 0.2, validation_data=(x_train, y_train))

    score = model.evaluate(x_test, y_test, verbose= 0)
    print(score)
    
def dense_model(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size= 0.2, shuffle= False)

    encoder = OneHotEncoder(categories='auto')
    y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()
    print(x_train.shape, y_train.shape)

    model = Sequential([
        Flatten(input_shape=(20, 30)),
        Dense(600, activation= "relu"),
        Dense(300, activation= "relu"),
        Dense(100, activation= "relu"),
        Dense(50, activation= "relu"),
        Dense(3, activation= "softmax"),
    ])

    model.compile(loss= "categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size= 10, epochs = 30, verbose= 1, validation_split = 0.2, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose= 0)
    print(score)    


def data_to_date_font(psw, interval_date:int=20):
    df = read_mysql(password= psw)
    features = pd.DataFrame()
    label = pd.DataFrame()
    select_columns = ["date", "Trading_volume", 'Lowest_price', 'Highest_price', 'Closing_price', 'market_trading_value',
                      'weighted_Index', 'Dividend_Yield(%)', 'Price-to-Earnings_Ratio',
                      'Price_to_Book_Ratio', 'quantity_margin', 'quantity_borrowed',
                      'Foreign_Investor_Net_Buy', 'Investment_Net_Buy', 'Dealer_Net_Buy',
                      'Net_Buys'
                      ]
    label_column = ['stock_feature']
    analyze_columns = select_columns[1:]
    features[select_columns] = df[select_columns][20:-5].reset_index(drop=True)
    for i in analyze_columns:
        for j in range(0, interval_date):
            features[f'{i}_{j+1}'] = df[i].values[interval_date:-5]-df[i].values[interval_date-j-1:-j-1-5]
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # statistics hell
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    tmp = pd.DataFrame()
    # 計算移動平均線（Moving Average）
    df['MA'] = df['Closing_price'].rolling(window=5).mean()
    features['MA'] = df['MA'].values[20:-5]

    # 計算相對強弱指標（Relative Strength Index，RSI）
    window = 14
    delta = df['Closing_price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=window).mean()
    average_loss = loss.rolling(window=window).mean()
    rs = average_gain / average_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    features['RSI'] = df['RSI'].values[20:-5]
    
    # 計算隨機震盪指標（Stochastic Oscillator）
    n = 14
    lowest_low = df['Lowest_price'].rolling(window=n).min()
    highest_high = df['Highest_price'].rolling(window=n).max()
    df['%K'] = 100 * (df['Closing_price'] - lowest_low) / (highest_high - lowest_low)
    df['%D'] = df['%K'].rolling(window=3).mean()
    features['%D'] = df['%D'].values[20:-5]

    # 計算威廉指標（Williams %R）
    df['%R'] = -100 * (highest_high - df['Closing_price']) / (highest_high - lowest_low)
    features['%R'] = df['%R'].values[20:-5]

    # 計算MACD指標（Moving Average Convergence Divergence）
    short_ema = df['Closing_price'].ewm(span=12, adjust=False).mean()
    long_ema = df['Closing_price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    features['MACD'] = df['MACD'].values[20:-5]
    
    # 計算布林通道指標（Bollinger Bands）
    std = df['Closing_price'].rolling(window=20).std()
    df['UpperBand'] = df['MA'] + 2 * std
    df['LowerBand'] = df['MA'] - 2 * std
    features['UpperBand'] = df['UpperBand'].values[20:-5]
    features['LowerBand'] = df['LowerBand'].values[20:-5]

    # 計算成交量指標（Volume）
    df['Volume MA'] = df['Trading_volume'].rolling(window=5).mean()
    features['Volume_MA'] = df['Volume MA'].values[20:-5]
    
    # 計算黃金交叉和死亡交叉（Golden Cross and Death Cross）
    short_ma = df['Closing_price'].rolling(window=50).mean()
    long_ma = df['Closing_price'].rolling(window=200).mean()
    df['GoldenCross'] = np.where(short_ma > long_ma, 1, 0)
    df['DeathCross'] = np.where(short_ma < long_ma, 1, 0)
    features['GoldenCross'] = df['GoldenCross'].values[20:-5]
    features['DeathCross'] = df['DeathCross'].values[20:-5]
    
    # 順勢指標（Commodity Channel Index，CCI）
    tmp.drop(tmp.index, inplace=True)
    tmp['Typical Price'] = (df['Highest_price'] + df['Lowest_price'] + df['Closing_price']) / 3
    moving_average_period = 20
    tmp['Moving Average'] = tmp['Typical Price'].rolling(moving_average_period).mean()
    tmp['Mean Deviation'] = np.abs(tmp['Typical Price'] - tmp['Moving Average'])
    cci_constant = 0.015
    df['CCI'] = (tmp['Typical Price'] - tmp['Moving Average']) / (cci_constant * tmp['Mean Deviation'])
    features['CCI'] = df['CCI'].values[20:-5]
    ##################################################################
    
    label[label_column[0]] = df[label_column][20:-5].reset_index(drop=True)           
    return features, label

features, labels = data_to_date_font(psw)
features = np.asarray(features)
labels = np.asarray(labels)
x_treain, x_test, y_train, y_test = train_test_split(features, labels, test_size= 0.1, shuffle= False)
y_train =to_categorical(y_train)
y_test =to_categorical(y_test)

model = Sequential([
    Dense(features.shape[1], activation="relu", input_shape= (features.shape[1],)),
    BatchNormalization(),
    Dense(270, activation= "relu"),
    BatchNormalization(),
    Dropout(0.1),
    BatchNormalization(),
    Dense(250, activation= "relu"),
    BatchNormalization(),
    Dropout(0.1),
    Dense(200, activation= "relu"),
    BatchNormalization(),
    Dropout(0.1),
    Dense(100, activation= "relu"),
    BatchNormalization(),
    Dropout(0.1),
    Dense(100, activation= "relu"),
    BatchNormalization(),
    Dropout(0.1),
    Dense(80, activation= "relu"),
    BatchNormalization(),
    Dropout(0.1),
    Dense(70, activation= "relu"),
    Dense(50, activation= "sigmoid"),
    Dense(10, activation= "sigmoid"),
    Dense(3, activation= "softmax")
])

model.compile(loss= "categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

model.fit(x_treain, y_train, batch_size= 10, epochs= 500, verbose= 1,
          validation_data= (x_test, y_test))
score = model.evaluate(x_test, y_test)
print(score)






'''
['date', 'Trading_volume', 'Trading_value', 'Opening_price',
       'Highest_price', 'Lowest_price', 'Closing_price', 'Price_difference',
       'Number_of_trades', 'market_trading_value', 'weighted_Index',
       'market_point_change', 'Dividend_Yield(%)', 'Price-to-Earnings_Ratio',
       'Price_to_Book_Ratio', 'quantity_margin', 'amount_margin',
       'quantity_borrowed', 'amount_borrowed', 'Foreign_Investor_Buy',
       'Foreign_Investor_Sell', 'Foreign_Investor_Net_Buy',
       'Foreign_Proprietary_Buy', 'Foreign_Proprietary_Sell',
       'Foreign_Proprietary_Net_Buy', 'Investment_Buy', 'Investment_Sell',
       'Investment_Net_Buy', 'Dealer_Net_Buy', 'Dealer_Buy(proprietary)',
       'Dealer_Sell(proprietary)', 'Dealer_Net Buy(proprietary)',
       'Dealer_Buy(hedging)', 'Dealer_Sell(hedging)',
       'Dealer_Net Buy(hedging)', 'Net_Buys', 'market_feature',
       'stock_feature']
'''
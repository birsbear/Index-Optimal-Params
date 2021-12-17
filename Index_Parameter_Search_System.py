# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 16:04:29 2021

@author: tis05
"""
import msvcrt
import sqlite3, sys, json, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client


class strategy:
    def __init__(self):
        self.win_count = {"L/S":[0,0] , "L":[0,0] , "S":[0,0]}
        self.win_rate = {"L/S":0 , "L":0 , "S":0}
        self.profit = {"L/S":0 , "L":0 , "S":0}
        self.profit_factor = {"L/S":0 , "L":0 , "S":0}
        self.RTP = {"L/S":0 , "L":0 , "S":0}

class DBsystem:
    def __init__(self, name, path = './'):
        self.db_name = name+'.db'
        self.db_path = path + self.db_name
        self.con = sqlite3.connect(self.db_path)
        self.cursor = self.con.cursor()
        pass
    
    def create_table(self,table_name, primary_key, primary_type, columns_name, columns_type):
        execute = "CREATE TABLE IF NOT EXISTS {table_name} ( {primary_key} {primary_type} PRIMARY KEY NOT NULL".format(table_name = table_name, primary_key = primary_key, primary_type = primary_type)
        
        for i in range(len(columns_name)):
            execute += ", {column_name} {column_type} NOT NULL".format(column_name = columns_name[i], column_type = columns_type[i])
        execute += ");"
        # print(execute)
        self.cursor.execute('''{}'''.format(execute))
        self.con.commit()

    def insert_data(self, table_name, columns_name, columns_value):
        execute = "INSERT INTO {} ({}) VALUES ({});".format(table_name, ','.join(columns_name), ', '.join(columns_value))
        self.cursor.execute('''{}'''.format(execute))
        self.con.commit()
        


class Order:
    def __init__(self, symbol="BTCUSDT", amount=0, open_price=0,order_type='Long', close_price = 0, order_date='', load_order = False, order_str = ''):
        if load_order:
            self.load_json(order_str)
        else:
            self.symbol = symbol
            self.amount = amount
            self.open_price = open_price
            self.close_price = close_price
            self.principal = self.amount*self.open_price
            self.type = order_type
            self.state = True
            self.order_date = order_date
            self.max_profit = 0
            self.max_drawdown = 0
            self.max_profit_time = ''
            self.max_drawdown_time = ''
    def close(self, close_price):
        self.close_price = close_price
        if self.type == 'Long':
            self.profit = (self.close_price - self.open_price)*self.amount
            self.rtp = (self.profit+self.principal)/self.principal
        elif self.type == 'Short':
            self.profit = (self.open_price - self.close_price)*self.amount
            self.rtp = (self.profit+self.principal)/self.principal
        self.state = False

    def save_json(self):
        if self.state:
            self.order_dict = {'symbol' : self.symbol,
                          'amount' : self.amount,
                          'open_price' : self.open_price,
                          'close_price' : self.close_price,
                          'principal' : self.principal,
                          'type' : self.type,
                          'state' : self.state,
                          }
        else:
            self.order_dict = {'symbol' : self.symbol,
                          'amount' : self.amount,
                          'open_price' : self.open_price,
                          'close_price' : self.close_price,
                          'principal' : self.principal,
                          'type' : self.type,
                          'state' : self.state,
                          'profit' : self.profit,
                          'rtp' : self.rtp
                          }
        json_str = json.dumps(self.order_dict)
        return json_str

    def load_json(self, json_str):
        self.order_dict = json.loads(json_str)
        self.symbol = self.order_dict['symbol']
        self.amount = self.order_dict['amount']
        self.open_price = self.order_dict['open_price']
        self.close_price = self.order_dict['close_price']
        self.principal = self.order_dict['principal']
        self.type = self.order_dict['type']
        self.state = self.order_dict['state']
        if self.state == False:
            self.profit = self.order_dict['profit']
            self.rtp = self.order_dict['rtp']
            
    def update(self, high, low, date):
        
        if self.type == 'Long':
            profit = (high-self.open_price)/self.open_price if high != self.open_price else 0
            drawdown = (self.open_price-low)/self.open_price if high != self.open_price else 0
        elif self.type == 'Short':
            profit = (self.open_price-low)/self.open_price if high != self.open_price else 0
            drawdown = (high-self.open_price)/self.open_price if high != self.open_price else 0
        if  profit > self.max_profit:
            self.max_profit = profit
            self.max_profit_time = str(datetime.fromisoformat(date)+timedelta(hours = 8))
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_time = str(datetime.fromisoformat(date)+timedelta(hours = 8))
        
class Wallet:
    def __init__(self):
        self.one_oder_money = 100000
        self.money = 0
        self.order_list = []
        self.order_history = []
        
        self.long_win = 0
        self.long_wcount = 0
        self.long_lose = 0
        self.long_lcount = 0
        
        self.short_win = 0
        self.short_wcount = 0
        self.short_lose = 0
        self.short_lcount = 0
        
        
        self.last_order_wl = 0          #連續輸贏狀態
        self.temp_continue_trades = 0   #連續次數
        self.temp_continue_money = 0    #連續金額
        self.temp_time = ['','']        #持續期間
        
        self.max_loss = 0               #最大虧損
        self.max_drawdown = 0           #最大連續虧損
        self.max_drawdown_trades = 0    #最大連續虧損次數
        self.drawdown_date = ['','']    #虧損持續期間
        self.max_win = 0                #最大收益
        self.max_profit = 0             #最大連續收益
        self.max_profit_trades = 0      #最大連續收益次數
        self.profit_date = ['','']      #收益持續期間
        
        self.once_max_profit = 0
        self.once_max_profit_time = ''
        self.once_max_drawdown = 0
        self.once_max_drawdown_time = ''
    
    def order_update(self, high, low, date):
        if self.order_list != []:
            order = self.order_list[-1]
            order.update(high, low, date)
    
    def wallet_update(self):
        
        order = self.order_history[-1]
        
        if order.profit <0:
            if order.profit < self.max_loss :
                self.max_loss = order.profit
            if self.last_order_wl < 0:
                self.last_order_wl -= 1
                self.temp_continue_money += order.profit
                self.temp_continue_trades += 1
            else:
                self.last_order_wl = -1
                self.temp_time[1] = str(datetime.fromisoformat(order.order_date)+timedelta(hours = 8))
                self.wl_state()
                self.temp_continue_money += order.profit
                self.temp_continue_trades += 1
                self.temp_time[0] = str(datetime.fromisoformat(order.order_date)+timedelta(hours = 8))
        elif order.profit >= 0:
            if order.profit > self.max_win:
                self.max_win = order.profit
                
            if self.last_order_wl > 0:
                self.last_order_wl += 1
                self.temp_continue_money += order.profit
                self.temp_continue_trades += 1
            else:
                self.last_order_wl = 1
                self.temp_time[1] = str(datetime.fromisoformat(order.order_date)+timedelta(hours = 8))
                self.wl_state()
                self.temp_continue_money += order.profit
                self.temp_continue_trades += 1
                self.temp_time[0] = str(datetime.fromisoformat(order.order_date)+timedelta(hours = 8))
                
        if order.max_profit > self.once_max_profit:
            self.once_max_profit = order.max_profit
            self.once_max_profit_time = order.max_profit_time
        if order.max_drawdown > self.once_max_drawdown:
            self.once_max_drawdown = order.max_drawdown
            self.once_max_drawdown_time = order.max_drawdown_time
    def wl_state(self):
        
        if self.temp_continue_money < self.max_drawdown:
            self.max_drawdown = self.temp_continue_money
            self.max_drawdown_trades = self.temp_continue_trades
            self.drawdown_date = self.temp_time
        elif self.temp_continue_money > self.max_profit:
            self.max_profit = self.temp_continue_money
            self.max_profit_trades = self.temp_continue_trades
            self.profit_date = self.temp_time
        
        self.temp_continue_trades = 0   #連續次數
        self.temp_continue_money = 0    #連續金額
        self.temp_time = ['','']
    
    def avg_max_wallet_profit_drawdown(self):
        
        long_profit_list = []
        long_drawdown_list = []
        short_profit_list = []
        short_drawdown_list = []
        total_profit_list = []
        total_drawdown_list = []
        
        for o in self.order_history:
            if o.type == 'Long':
                long_profit_list.append(o.max_profit)
                long_drawdown_list.append(o.max_drawdown)
            elif o.type == 'Short':
                short_profit_list.append(o.max_profit)
                short_drawdown_list.append(o.max_drawdown)
            total_profit_list.append(o.max_profit)
            total_drawdown_list.append(o.max_drawdown)
            
        self.average_long_profit = np.average(long_profit_list) if long_profit_list != [] else 0
        self.average_long_drawdown = np.average(long_drawdown_list) if long_drawdown_list != [] else 0
        self.average_short_profit = np.average(short_profit_list) if short_profit_list != [] else 0
        self.average_short_drawdown = np.average(short_drawdown_list) if short_drawdown_list != [] else 0
        self.average_profit = np.average(total_profit_list) if total_profit_list != [] else 0
        self.average_drawdown = np.average(total_drawdown_list) if total_drawdown_list != [] else 0
        
        
class Cryptocurrency:

    def __init__(self, symbol, db):
        self.symbol = symbol[1:]+"USDT"if "USDT" not in symbol else symbol[1:]
        self.db = db
        self.k_lines = {'1d':[], '12h':[], '4h':[], '2h':[], '1h':[], '30m':[]}
        self.db_k_lines = {'1d':[], '12h':[], '4h':[], '2h':[], '1h':[], '30m':[]}
        self.strategy_db = {'1d':[], '12h':[], '4h':[], '2h':[], '1h':[], '30m':[]}
        
            
    def update_db(self):
        
        for k_type in self.k_lines.keys():
            self.check_db(k_type)
            self.update_k_lines(k_type)
            
    def check_db(self, k_line_type):
        table = '_'+self.symbol+'_'+k_line_type
        execute = "SELECT count(*) FROM sqlite_master where type='table' AND name = '{}'".format(table)
        self.db.cursor.execute(execute)
        feedback = self.db.cursor.fetchall()[0][0]
        if feedback == 0:
            self.create_kline_db(table)
            
        self.db_k_lines[k_line_type] = pd.read_sql("SELECT * FROM {}".format(table), self.db.con)
        
    def check_strategy_db(self, k_line_type, strategy_name):
        table = f'_{self.symbol}_{kline_type}_{strategy}' 
        feedback = self.db.check(table)
        if feedback == 0:
            if strategy_name == 'SuperTrend':
                self.create_supertrend_table(table)
            elif strategy_name == "KDJ":
                self.create_kdj_table(table)
        
        self.strategy_db[k_line_type] = pd.read_sql(f"SELECT * FROM {table}", self.db.con)
        
    def create_kline_db(self, table):
        _, symbol, k_line_type = table.split('_')
        self.db.create_table(table, 
                        "OpenTime",
                        "TEXT",
                        ['Open','High','Low','Close','Volumn','CloseTime','Quote_asset_volume','Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume','Ignore'], 
                        ['REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'INT', 'REAL', 'INT', 'REAL', 'REAL', 'REAL']
                        )
    
    
    def update_k_lines(self, k_line_type):
        table = '_'+self.symbol+'_'+k_line_type
        if self.db_k_lines[k_line_type].shape[0] > 1:
            start_time = self.db_k_lines[k_line_type].OpenTime.iloc[-1]
            # delete_time = self.db_k_lines[k_line_type].OpenTime.iloc[-1]
            self.db.cursor.execute("""DELETE from {} where {} = '{}' """.format(table, 'OpenTime', start_time))
            self.db.con.commit()
        else:
            start_time = '2017-08-17 00:00:00'
        # print(start_time, k_line_type)
        self.k_lines[k_line_type] = self.get_k_llines(k_line_type, start_time = start_time)
        self.k_lines[k_line_type].to_sql(table, self.db.con, if_exists='append', index = False)
            
    def get_k_llines(self, k, start_time = '2017-08-17 00:00:00', end_time = None):
        retry = 0
        while retry < 6:
            try:
                if end_time == None:
                    k_lines = pd.DataFrame(client.get_historical_klines(self.symbol, k, start_time), columns=(['OpenTime','Open','High','Low','Close','Volumn','CloseTime','Quote_asset_volume','Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume','Ignore']))
                else:
                    k_lines = pd.DataFrame(client.get_historical_klines(self.symbol, k, start_time, end_time), columns=(['OpenTime','Open','High','Low','Close','Volumn','CloseTime','Quote_asset_volume','Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume','Ignore']))
                    
                k_lines[['OpenTime']] = k_lines.OpenTime.apply(lambda x : str(datetime.fromtimestamp((x-28800000)/1000)))
                
                k_lines[['Open', 'High','Low','Close','Volumn']] = k_lines[['Open', 'High','Low','Close','Volumn']].astype(float)
                retry = 6
            except Exception as e:
                retry += 1
                print("Retry get {} kline data...{}/5".format(self.symbol, retry))
                
            # sys.exit()
        return k_lines
    
    def get_db_last_date(self):
        self.check_db('1d')
        if self.db_k_lines['1d'].shape[0] == 0:
            print("No K-line in DB")
        else:
            print("最後一筆日K資料為 {}".format(self.db_k_lines['1d'].OpenTime.iloc[-1]))
            
    def create_supertrend_table(self, table):
        _, symbol, k_line_type, _ = table.split('_')
        
        column_names = ['Open','High','Low','Close','tr','souce1','souce2','souce3','souce4']
        
        for period in range(1,8):
            column_names.append(f"atr_p{period}")
            for source in range(1,5):
                column_names.append(f"up_p{period}_s{source}")
                column_names.append(f"down_p{period}_s{source}")
                column_names.append(f"trend_p{period}_s{source}")
                
        column_type = ['REAL']*len(column_names)
        
        self.db.create_table(table, 
                        "OpenTime",
                        "TEXT",
                        column_names, 
                        column_type
                        )
        pass
    
    def create_kdj_table(self, table):
        _, symbol, k_line_type, _ = table.split('_')
        
        column_names = ['Open','High','Low','Close','rsv','pK','pD','pJ','trend']
        
        for ilong in range(1,101):
            column_names.append(f"rsv_l{ilong}")
            column_names.append(f"pK_l{ilong}")
            column_names.append(f"pD_l{ilong}")
            column_names.append(f"pJ_l{ilong}")
            column_names.append(f"trend_l{ilong}")
                
        column_type = ['REAL']*len(column_names)
        
        self.db.create_table(table, 
                        "OpenTime",
                        "TEXT",
                        column_names, 
                        column_type
                        )
        pass
    
    def SuperTrend(self, k_line_type, source, period, factor):
        data_table = '_' + self.symbol+ '_' + k_line_type 
        strategy_table = data_table + '_SuperTrend'
        df = self.db_k_lines[k_line_type]
        n_array = df[['OpenTime', 'High', 'Low', 'Open', 'Close']].to_numpy()
        hl, hc, cl = n_array[1:,1]-n_array[1:,2] , abs(n_array[1:,1]-n_array[:-1,4]), abs(n_array[1:,2]-n_array[:-1,4])
        tr = np.insert(np.max([hl,hc,cl], axis=0),0,n_array[0,1]-n_array[0,2])
        if source == 'close':
            src = n_array[:,4]
        elif source == "hl2":
            src = (n_array[:,1]+n_array[:,2])/2
        elif source == "hlc3":
            src = (n_array[:,1]+n_array[:,2]+n_array[:,4])/3
        elif source == "ohlc4":
            src = (n_array[:,1]+n_array[:,2]+n_array[:,3]+n_array[:,4])/4
        trend, p1atr, p1up, p1down = np.ones((n_array.shape[0],)),np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],))
        new_n = np.c_[n_array,tr,src]
        p_array = np.c_[trend,p1atr,p1up,p1down,p1down].astype(object)
        for ind, row in enumerate(new_n):
            if ind >= period:
                last_row = new_n[ind-1]
                if ind == period:
                    atr = new_n[ind-period+1 : ind+1,5].sum()/period
                else:
                    alpha = 1/period
                    atr = row[5]*alpha + (1-alpha)*p_array[ind-1,1]
                p_array[ind,1] = atr
                up = row[6] - factor*atr
                up1 = p_array[ind-1,2]
                up = max(up,up1) if last_row[4] > up1 else up
                p_array[ind,2] = up
                
                down = row[6] + factor*atr
                down1 = p_array[ind-1,3]
                down = min(down,down1) if last_row[4] < down1 else down
                p_array[ind,3] = down
                
                t_ = p_array[ind-1,0]
                if (t_ == -1) & (row[4] > down1):
                    t_ = 1
                elif (t_ == 1) & (row[4] < up1):
                    t_ = -1
                p_array[ind,0] = t_
        new_n = np.c_[new_n, p_array]
        return new_n
    
    def SuperTrend_Parameter(self,k_line_type,is_save):
        data_table = '_' + self.symbol+ '_' + k_line_type 
        strategy_table = data_table + '_SuperTrend'
        df = self.db_k_lines[k_line_type]
        n_array = df[['OpenTime', 'High', 'Low', 'Open', 'Close']].to_numpy()
        hl, hc, cl = n_array[1:,1]-n_array[1:,2] , abs(n_array[1:,1]-n_array[:-1,4]), abs(n_array[1:,2]-n_array[:-1,4])
        tr = np.insert(np.max([hl,hc,cl], axis=0),0,n_array[0,1]-n_array[0,2])
        src1, src2, src3, src4 = n_array[:,4] , (n_array[:,1]+n_array[:,2])/2, (n_array[:,1]+n_array[:,2]+n_array[:,4])/3, (n_array[:,1]+n_array[:,2]+n_array[:,3]+n_array[:,4])/4
        trend, p1atr, p1up, p1down = np.ones((n_array.shape[0],)),np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],))
        new_n = np.c_[n_array,tr,src1,src2,src3,src4]
        
        result = {"Source":[],"Param":[],
                  "RTP_LS":[], "RTP_L":[], "RTP_S":[], 
                  "ProfitFactor_LS":[], "ProfitFactor_L":[], "ProfitFactor_S":[], 
                  "Profit_LS":[], "Profit_L":[], "Profit_S":[],
                  "WinRate_LS":[], "WinRate_L":[], "WinRate_S":[],
                  "WinCount_LS":[], "WinCount_L":[], "WinCount_S":[],
                  "Max_win":[], "Max_Profit":[], "Max_Profit_trades":[],"Profit_Date":[],
                  "Max_loss":[], "Max_Drawdown":[], "Max_Drawdown_trades":[], "Drawdown_Date":[],
                  "Once_Max_Profit":[], "Once_Max_Profit_time":[], "Once_Max_Drawdown":[], "Once_Max_Drawdown_time":[],
                  "Average_Long_Profit":[], "Average_Long_Drawdown":[], "Average_Short_Profit":[], "Average_Short_Drawdown":[], "Average_Total_Profit":[], "Average_Total_Drawdown":[]
                  }
        source = ["close", "hl2", "hlc3", "ohlc4"]
        factor = np.arange(0.1,10.1,0.1)
        factor_size = factor.shape[0]
        
        for period in range(1,8):
            for src in range(4):
                p_array = np.c_[trend,p1atr,p1up,p1down,p1down].astype(object)
                
                for i in range(new_n.shape[0]):
                    p_array[i,0] = np.ones((factor_size,))
                    p_array[i,2] = np.zeros((factor_size,))
                    p_array[i,3] = np.zeros((factor_size,))
                    # p_array[i,4] = np.array(['']*factor_size)
                buy_singal, sell_singal = np.zeros((factor_size,)), np.zeros((factor_size,))
                wallet_list = np.array([Wallet() for i in range(factor_size)])
                for ind, row in enumerate(new_n):
                    if ind >= period:
                        last_row = new_n[ind-1]
                        if ind == period:
                            atr = new_n[ind-period+1 : ind+1,5].sum()/period
                        else:
                            alpha = 1/period
                            atr = row[5]*alpha + (1-alpha)*p_array[ind-1,1]
                        p_array[ind,1] = atr
                
                        up = row[6+src] - factor*atr
                        up1 = p_array[ind-1,2]
                        x = np.where(last_row[4]>up1)
                        up[x] = np.max([up[x], up1[x]], axis =0)
                        p_array[ind,2] = up
                        
                        down = row[6+src] + factor*atr
                        down1 = p_array[ind-1,3]
                        y = np.where(last_row[4]<down1)
                        down[y] = np.min([down[y], down1[y]], axis =0)
                        p_array[ind,3] = down
                        t_ = np.array(p_array[ind-1,0])
                        t1 = np.where((t_ == -1) & (row[4] > down1))
                        t2 = np.where((t_ == 1) & (row[4] < up1))
                        t_[t1] = 1
                        t_[t2] = -1
                        p_array[ind,0] = t_
                        
                        
                        # price = float(row[3])
                        
                        # buy_order = np.where(buy_singal == 1)
                        # for b in buy_order[0]:
                        #     wallet = wallet_list[b]
                        #     if wallet.order_list != [] and wallet.order_list[0].type == 'Short':
                        #         wallet.order_list[0].close(price)
                        #         profit = wallet.order_list[0].profit
                        #         wallet.money += profit
                        #         history_order = wallet.order_list.pop()
                                
                        #         if profit >= 0:
                        #             wallet.short_win += profit
                        #             wallet.short_wcount += 1
                        #         else:
                        #             wallet.short_lose += profit
                        #             wallet.short_lcount += 1
                                
                        #         wallet.order_history.append(history_order)
                        #         wallet.wallet_update()
                            
                        #         # p_array[ind,4][b] = history_order.save_json()
                        #     amount = wallet.one_oder_money / price
                        #     wallet.order_list.append(Order(self.symbol, amount, price, "Long", order_date = row[0]))
                        # sell_order = np.where(sell_singal == 1)
                        # for s in sell_order[0]:
                        #     wallet = wallet_list[s]
                        #     if wallet.order_list != [] and wallet.order_list[0].type == 'Long':
                        #         wallet.order_list[0].close(price)
                        #         profit = wallet.order_list[0].profit
                        #         wallet.money += profit
                        #         history_order = wallet.order_list.pop()
                                
                        #         if profit >= 0:
                        #             wallet.long_win += profit
                        #             wallet.long_wcount += 1
                        #         else:
                        #             wallet.long_lose += profit
                        #             wallet.long_lcount += 1
                                
                        #         wallet.order_history.append(history_order)
                        #         wallet.wallet_update()
                            
                        #         # p_array[ind,4][s] = history_order.save_json()
                        #     amount = wallet.one_oder_money / price
                        #     wallet.order_list.append(Order(self.symbol, amount, price, "Short", order_date = row[0]))
                        
                        # for w_ in wallet_list:
                        #     w_.order_update(row[1], row[2], row[0])
                        # buy_singal, sell_singal = np.zeros(t_.shape), np.zeros(t_.shape)
                        
                        
                        
                        
                        
                        # buy_ = np.where(((p_array[ind-1,0]+p_array[ind,0]) == 0 )& (p_array[ind,0]==1))
                        # sell_ = np.where(((p_array[ind-1,0]+p_array[ind,0]) == 0 )& (p_array[ind,0]==-1))
                        # buy_singal[buy_], sell_singal[sell_] = 1, 1
                        
                        
                        
                        
                        
                # for ind, w in enumerate(wallet_list):
                    
                #     result["Source"].append("{}".format(source[src]))
                #     result["Param"].append("{:.2f}, {:.2f}".format(period,(ind+1)*0.1))
                    
                    
                #     result["ProfitFactor_LS"].append((w.long_win + w.short_win)/-1*(w.long_lose + w.short_lose) if (w.long_lose + w.short_lose) != 0 else (w.long_win + w.short_win))
                #     result["ProfitFactor_L"].append(w.long_win / -1*w.long_lose if w.long_lose != 0 else w.long_win)
                #     result["ProfitFactor_S"].append(w.short_win/ -1*w.short_lose if w.short_lose != 0 else w.short_win)
                #     ls,l,s = w.long_win + w.short_win + w.long_lose + w.short_lose, w.long_win + w.long_lose, w.short_win +w.short_lose
                #     lsw,lw,sw = [w.long_wcount+w.short_wcount, w.long_wcount+w.short_wcount+w.long_lcount+w.short_lcount], [w.long_wcount, w.long_wcount+w.long_lcount], [w.short_wcount, w.short_wcount+w.short_lcount]
                    
                #     w.avg_max_wallet_profit_drawdown()
                        
            
                #     result["Profit_LS"].append(ls)
                #     result["Profit_L"].append(l)
                #     result["Profit_S"].append(s)
                #     result["RTP_LS"].append(ls/w.one_oder_money)
                #     result["RTP_L"].append(l/w.one_oder_money)
                #     result["RTP_S"].append(s/w.one_oder_money)
                #     result["WinRate_LS"].append(lsw[0]/lsw[1] if lsw[1] != 0 else lsw[0])
                #     result["WinRate_L"].append(lw[0]/lw[1] if lw[1] != 0 else lw[0])
                #     result["WinRate_S"].append(sw[0]/sw[1] if sw[1] != 0 else sw[0])
                #     result["WinCount_LS"].append(lsw)
                #     result["WinCount_L"].append(lw)
                #     result["WinCount_S"].append(sw)
            
                #     result["Max_win"].append(w.max_win)
                #     result["Max_Profit"].append(w.max_profit)
                #     result["Max_Profit_trades"].append(w.max_profit_trades)
                #     result["Profit_Date"].append(w.profit_date)
                    
                #     result["Max_loss"].append(w.max_loss)
                #     result["Max_Drawdown"].append(w.max_drawdown)
                #     result["Max_Drawdown_trades"].append(w.max_drawdown_trades)
                #     result["Drawdown_Date"].append(w.drawdown_date)
                    
                #     result['Once_Max_Profit'].append(w.once_max_profit*100)
                #     result['Once_Max_Profit_time'].append(w.once_max_profit_time)
                #     result['Once_Max_Drawdown'].append(w.once_max_drawdown*100)
                #     result['Once_Max_Drawdown_time'].append(w.once_max_drawdown_time)
                    
                #     result['Average_Long_Profit'].append(w.average_long_profit*100)
                #     result['Average_Long_Drawdown'].append(w.average_long_drawdown*100)
                #     result['Average_Short_Profit'].append(w.average_short_profit*100)
                #     result['Average_Short_Drawdown'].append(w.average_short_drawdown*100)
                #     result['Average_Total_Profit'].append(w.average_profit*100)
                #     result['Average_Total_Drawdown'].append(w.average_drawdown*100)
                    
                            
                new_n = np.c_[new_n, p_array]
        # self.strategy_reback_result = pd.DataFrame(result)
        # if is_save:
        #     self.strategy_reback_result.to_csv("./BackTest/SuperTrend/{}/{}_{}.csv".format(k_line_type,self.symbol,k_line_type), index = 0)
        return new_n
        
    
    def SuperTrend_Parameter_new(self,k_line_type,is_save):
        data_table = '_' + self.symbol+ '_' + k_line_type 
        strategy_table = data_table + '_SuperTrend'
        df = self.db_k_lines[k_line_type]
        n_array = df[['OpenTime', 'Open', 'High', 'Low', 'Close']].to_numpy()
        hl, hc, cl = n_array[1:,2]-n_array[1:,3] , abs(n_array[1:,2]-n_array[:-1,4]), abs(n_array[1:,3]-n_array[:-1,4])
        tr = np.insert(np.max([hl,hc,cl], axis=0),0,n_array[0,2]-n_array[0,3])
        src1, src2, src3, src4 = n_array[:,4] , (n_array[:,2]+n_array[:,3])/2, (n_array[:,2]+n_array[:,3]+n_array[:,4])/3, (n_array[:,1]+n_array[:,2]+n_array[:,3]+n_array[:,4])/4
        trend_s1, trend_s2, trend_s3, trend_s4, =  np.ones((n_array.shape[0],)), np.ones((n_array.shape[0],)), np.ones((n_array.shape[0],)), np.ones((n_array.shape[0],))
        p1atr, up_s1, down_s1, up_s2, down_s2, up_s3, down_s3, up_s4, down_s4 =np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],)),np.zeros((n_array.shape[0],))
        new_n = np.c_[n_array,tr,src1,src2,src3,src4]
        
        result = {"Source":[],"Param":[],
                  "RTP_LS":[], "RTP_L":[], "RTP_S":[], 
                  "ProfitFactor_LS":[], "ProfitFactor_L":[], "ProfitFactor_S":[], 
                  "Profit_LS":[], "Profit_L":[], "Profit_S":[],
                  "WinRate_LS":[], "WinRate_L":[], "WinRate_S":[],
                  "WinCount_LS":[], "WinCount_L":[], "WinCount_S":[],
                  "Max_win":[], "Max_Profit":[], "Max_Profit_trades":[],"Profit_Date":[],
                  "Max_loss":[], "Max_Drawdown":[], "Max_Drawdown_trades":[], "Drawdown_Date":[],
                  "Once_Max_Profit":[], "Once_Max_Profit_time":[], "Once_Max_Drawdown":[], "Once_Max_Drawdown_time":[],
                  "Average_Long_Profit":[], "Average_Long_Drawdown":[], "Average_Short_Profit":[], "Average_Short_Drawdown":[], "Average_Total_Profit":[], "Average_Total_Drawdown":[]
                  }
        source = ["close", "hl2", "hlc3", "ohlc4"]
        factor = np.arange(0.1,10.1,0.1)
        factor_size = factor.shape[0]
        
        for period in range(1,8):
            p_array = np.c_[p1atr,up_s1,down_s1,trend_s1,up_s2,down_s2,trend_s2,up_s3,down_s3,trend_s3,up_s4,down_s4,trend_s4].astype(object)
            
            for i in range(new_n.shape[0]):
                for j in [1,2,4,5,7,8,10,11]:
                    p_array[i,j] = np.zeros((factor_size,))
                for j in [3,6,9,12]:
                    p_array[i,j] = np.ones((factor_size,))
                    
            buy_singal, sell_singal = np.zeros((factor_size,)), np.zeros((factor_size,))
            wallet_list = np.array([Wallet() for i in range(factor_size)])
            for ind, row in enumerate(new_n):
                if ind >= period:
                    last_row = new_n[ind-1]
                    if new_n[ind-1,5] == 0.0:
                        atr = new_n[ind-period+1 : ind+1,5].sum()/period
                    else:
                        alpha = 1/period
                        atr = row[5]*alpha + (1-alpha)*p_array[ind-1,0]
                    p_array[ind,0] = atr
            
                    up_1 = row[6] - factor*atr
                    up_2 = row[7] - factor*atr
                    up_3 = row[8] - factor*atr
                    up_4 = row[9] - factor*atr
                    
                    up1_1 = p_array[ind-1,1]
                    up1_2 = p_array[ind-1,4]
                    up1_3 = p_array[ind-1,7]
                    up1_4 = p_array[ind-1,10]
                    
                    x_1 = np.where(last_row[4]>up1_1)
                    x_2 = np.where(last_row[4]>up1_2)
                    x_3 = np.where(last_row[4]>up1_3)
                    x_4 = np.where(last_row[4]>up1_4)
                    
                    up_1[x_1] = np.max([up_1[x_1], up1_1[x_1]], axis =0)
                    up_2[x_2] = np.max([up_2[x_2], up1_2[x_2]], axis =0)
                    up_3[x_3] = np.max([up_3[x_3], up1_3[x_3]], axis =0)
                    up_4[x_4] = np.max([up_4[x_4], up1_4[x_4]], axis =0)
                    
                    p_array[ind,1] = up_1
                    p_array[ind,4] = up_2
                    p_array[ind,7] = up_3
                    p_array[ind,10] = up_4
                    
                    down_1 = row[6] + factor*atr
                    down_2 = row[7] + factor*atr
                    down_3 = row[8] + factor*atr
                    down_4 = row[9] + factor*atr
                    
                    down1_1 = p_array[ind-1,2]
                    down1_2 = p_array[ind-1,5]
                    down1_3 = p_array[ind-1,8]
                    down1_4 = p_array[ind-1,11]
                    
                    y_1 = np.where(last_row[4]<down1_1)
                    y_2 = np.where(last_row[4]<down1_2)
                    y_3 = np.where(last_row[4]<down1_3)
                    y_4 = np.where(last_row[4]<down1_4)
                    
                    down_1[y_1] = np.min([down_1[y_1], down1_1[y_1]], axis =0)
                    down_2[y_2] = np.min([down_2[y_2], down1_2[y_2]], axis =0)
                    down_3[y_3] = np.min([down_3[y_3], down1_3[y_3]], axis =0)
                    down_4[y_4] = np.min([down_4[y_4], down1_4[y_4]], axis =0)
                    
                    p_array[ind,2] = down_1
                    p_array[ind,5] = down_2
                    p_array[ind,8] = down_3
                    p_array[ind,11] = down_4
                    
                    
                    t_1 = np.array(p_array[ind-1,3])
                    t_2 = np.array(p_array[ind-1,6])
                    t_3 = np.array(p_array[ind-1,9])
                    t_4 = np.array(p_array[ind-1,12])
                    
                    t1_1 = np.where((t_1 == -1) & (row[4] > down1_1))
                    t1_2 = np.where((t_2 == -1) & (row[4] > down1_2))
                    t1_3 = np.where((t_3 == -1) & (row[4] > down1_3))
                    t1_4 = np.where((t_4 == -1) & (row[4] > down1_4))
                    
                    t2_1 = np.where((t_1 == 1) & (row[4] < up1_1))
                    t2_2 = np.where((t_2 == 1) & (row[4] < up1_2))
                    t2_3 = np.where((t_3 == 1) & (row[4] < up1_3))
                    t2_4 = np.where((t_4 == 1) & (row[4] < up1_4))
                    
                    
                    t_1[t1_1] = 1
                    t_2[t1_2] = 1
                    t_3[t1_3] = 1
                    t_4[t1_4] = 1
                    
                    
                    t_1[t2_1] = -1
                    t_2[t2_2] = -1
                    t_3[t2_3] = -1
                    t_4[t2_4] = -1
                    
                    p_array[ind,3] = t_1
                    p_array[ind,6] = t_2
                    p_array[ind,9] = t_3
                    p_array[ind,12] = t_4
                            
            new_n = np.c_[new_n, p_array]
        return new_n
    
    def KDJ_Parameter(self, k_line_type,is_save):
        data_table = '_' + self.symbol+ '_' + k_line_type 
        strategy_table = data_table + '_KDJ'
        result = {"Param":[],
                  "RTP_LS":[], "RTP_L":[], "RTP_S":[], 
                  "ProfitFactor_LS":[], "ProfitFactor_L":[], "ProfitFactor_S":[],
                  "Profit_LS":[], "Profit_L":[], "Profit_S":[],
                  "WinRate_LS":[], "WinRate_L":[], "WinRate_S":[],
                  "WinCount_LS":[], "WinCount_L":[], "WinCount_S":[],
                  "Max_win":[], "Max_Profit":[], "Max_Profit_trades":[],"Profit_Date":[],
                  "Max_loss":[], "Max_Drawdown":[], "Max_Drawdown_trades":[], "Drawdown_Date":[],
                  "Once_Max_Profit":[], "Once_Max_Profit_time":[], "Once_Max_Drawdown":[], "Once_Max_Drawdown_time":[],
                  "Average_Long_Profit":[], "Average_Long_Drawdown":[], "Average_Short_Profit":[], "Average_Short_Drawdown":[], "Average_Total_Profit":[], "Average_Total_Drawdown":[]
                  
                  }
        
        
        for ilong in range(1,100):
            wallet_list = np.array([Wallet() for i in range(1,100)])
            n = self.db_k_lines[k_line_type][['Open','High','Low','Close']].to_numpy()
            isig = np.array(range(1,100))
            m = 1
            rsv_list = np.zeros(n[:,1].shape)
            trend = np.ones((rsv_list.shape[0],99)).astype(object)
            buy_singal, sell_singal = np.zeros((99,)), np.zeros((99,))
            k_list = np.zeros((rsv_list.shape[0],99)).astype(object)
            d_list = np.zeros((rsv_list.shape[0],99)).astype(object)
            j_list = np.zeros((rsv_list.shape[0],99)).astype(object)
            for ind,row in enumerate(n):
                if ind < ilong-1:
                    continue
                c2 = row[3]
                h2 = np.max(n[ind-ilong+1:ind+1,1]) 
                l2 = np.min(n[ind-ilong+1:ind+1,2]) 
                
                if h2 == l2:
                    rsv_list[ind] = rsv_list[ind-1]
                else:
                    rsv_list[ind] = 100*((c2-l2)/(h2-l2))
        
                p1 = rsv_list[ind]*m
                p2 = isig-m
        
                p3 = k_list[ind-1,:] 
                k_list[ind,:] = (p1+p2*p3)/isig
        
        
        
                p1 = k_list[ind,:]*m
                p2 = isig-m
        
                p3 = d_list[ind-1,:]
                d_list[ind,:] = (p1+p2*p3)/isig
        
                j_list[ind,:] = 3*k_list[ind] -2*d_list[ind]
                
                
                
                price = row[0]
                
                
                buy_order = np.where(buy_singal == 1)
                for b in buy_order[0]:
                    wallet = wallet_list[b]
                    if wallet.order_list != [] and wallet.order_list[0].type == 'Short':
                        wallet.order_list[0].close(price)
                        profit = wallet.order_list[0].profit
                        wallet.money += profit
                        history_order = wallet.order_list.pop()
                        
                        if profit >= 0:
                            wallet.short_win += profit
                            wallet.short_wcount += 1
                        else:
                            wallet.short_lose += profit
                            wallet.short_lcount += 1
                        
                        wallet.order_history.append(history_order)
                        wallet.wallet_update()
                    
                    amount = wallet.one_oder_money / price
                    wallet.order_list.append(Order(self.symbol, amount, price, "Long", order_date = self.db_k_lines[k_line_type]['OpenTime'].iloc[ind]))
                sell_order = np.where(sell_singal == 1)
                for s in sell_order[0]:
                    wallet = wallet_list[s]
                    if wallet.order_list != [] and wallet.order_list[0].type == 'Long':
                        wallet.order_list[0].close(price)
                        profit = wallet.order_list[0].profit
                        wallet.money += profit
                        history_order = wallet.order_list.pop()
                        
                        if profit >= 0:
                            wallet.long_win += profit
                            wallet.long_wcount += 1
                        else:
                            wallet.long_lose += profit
                            wallet.long_lcount += 1
                        
                        wallet.order_history.append(history_order)
                        wallet.wallet_update()
                    
                    amount = wallet.one_oder_money / price
                    wallet.order_list.append(Order(self.symbol, amount, price, "Short", order_date = self.db_k_lines[k_line_type]['OpenTime'].iloc[ind]))
                
                for w_ in wallet_list:
                    w_.order_update(row[1], row[2], self.db_k_lines[k_line_type]['OpenTime'].iloc[ind])
                buy_singal, sell_singal = np.zeros((99,)), np.zeros((99,))
                
                
                
                long_x = np.where(j_list[ind,:] > d_list[ind,:])
                short_x = np.where(j_list[ind,:] < d_list[ind,:])
                no_x = np.where(j_list[ind,:] == d_list[ind,:])
                last_t = np.array(trend[ind-1, no_x])
                trend[ind,long_x], trend[ind,short_x], trend[ind,no_x] = 1, -1, last_t
                
                buy_ = np.where(((trend[ind-1,:]+trend[ind,:]) == 0 )& (trend[ind,:]==1))
                sell_ = np.where(((trend[ind-1,:]+trend[ind,:]) == 0 )& (trend[ind,:]==-1))
                buy_singal[buy_], sell_singal[sell_] = 1, 1
                
                
            
            for ind, w in enumerate(wallet_list):
                
                result["Param"].append('ilong:{:02}, isig:{:02}'.format(ilong,(ind+1)))
                
                
                result["ProfitFactor_LS"].append((w.long_win + w.short_win)/-1*(w.long_lose + w.short_lose) if (w.long_lose + w.short_lose) != 0 else (w.long_win + w.short_win))
                result["ProfitFactor_L"].append(w.long_win / -1*w.long_lose if w.long_lose != 0 else w.long_win)
                result["ProfitFactor_S"].append(w.short_win/ -1*w.short_lose if w.short_lose != 0 else w.short_win)
                ls,l,s = w.long_win + w.short_win + w.long_lose + w.short_lose, w.long_win + w.long_lose, w.short_win +w.short_lose
                lsw,lw,sw = [w.long_wcount+w.short_wcount, w.long_wcount+w.short_wcount+w.long_lcount+w.short_lcount], [w.long_wcount, w.long_wcount+w.long_lcount], [w.short_wcount, w.short_wcount+w.short_lcount]
        
        
                w.avg_max_wallet_profit_drawdown()
        
                result["Profit_LS"].append(ls)
                result["Profit_L"].append(l)
                result["Profit_S"].append(s)
                result["RTP_LS"].append(ls/w.one_oder_money)
                result["RTP_L"].append(l/w.one_oder_money)
                result["RTP_S"].append(s/w.one_oder_money)
                result["WinRate_LS"].append(lsw[0]/lsw[1] if lsw[1] != 0 else lsw[0])
                result["WinRate_L"].append(lw[0]/lw[1] if lw[1] != 0 else lw[0])
                result["WinRate_S"].append(sw[0]/sw[1] if sw[1] != 0 else sw[0])
                result["WinCount_LS"].append(lsw)
                result["WinCount_L"].append(lw)
                result["WinCount_S"].append(sw)
                
                result["Max_win"].append(w.max_win)
                result["Max_Profit"].append(w.max_profit)
                result["Max_Profit_trades"].append(w.max_profit_trades)
                result["Profit_Date"].append(w.profit_date)
                
                result["Max_loss"].append(w.max_loss)
                result["Max_Drawdown"].append(w.max_drawdown)
                result["Max_Drawdown_trades"].append(w.max_drawdown_trades)
                result["Drawdown_Date"].append(w.drawdown_date)
                    
                result['Once_Max_Profit'].append(w.once_max_profit*100)
                result['Once_Max_Profit_time'].append(w.once_max_profit_time)
                result['Once_Max_Drawdown'].append(w.once_max_drawdown*100)
                result['Once_Max_Drawdown_time'].append(w.once_max_drawdown_time)
                
                result['Average_Long_Profit'].append(w.average_long_profit*100)
                result['Average_Long_Drawdown'].append(w.average_long_drawdown*100)
                result['Average_Short_Profit'].append(w.average_short_profit*100)
                result['Average_Short_Drawdown'].append(w.average_short_drawdown*100)
                result['Average_Total_Profit'].append(w.average_profit*100)
                result['Average_Total_Drawdown'].append(w.average_drawdown*100)
                
                
                
        self.strategy_reback_result = pd.DataFrame(result)
        if is_save:
            self.strategy_reback_result.to_csv("./BackTest/KDJ/{}/{}_{}.csv".format(k_line_type,self.symbol,k_line_type), index = 0)
        

def update_k_lines_db(cyptor_list, db):
    cyptor_list = ['BTCUSDT','ETHUSDT']
    cyptor_numel = len(cyptor_list)
    for ind,symbol in enumerate(cyptor_list):
        print("\rUpdate {} K Line.....{}/{}".format(symbol, ind+1, cyptor_numel), end='')
        symbol = '_' + symbol
        coin = Cryptocurrency(symbol, db)
        coin.update_db()
        
def update_db_1m(cyptor_list, db):
    cyptor_numel = len(cyptor_list)
    for ind,symbol in enumerate(cyptor_list):
        print("\rUpdate {} K Line.....{}/{}".format(symbol, ind+1, cyptor_numel), end='')
        symbol = '_' + symbol
        coin = Cryptocurrency(symbol, db)
        coin.k_lines = {'1m':[]}
        coin.db_k_lines = {'1m':[]}
        coin.strategy_db = {'1m':[]}
        coin.update_db()
        
def parameter_search(symbol_list, index_name, k_line_type, is_save, db):
    
    for symbol in symbol_list:
        table = '_'+symbol+'_'+k_line_type+'_'+index_name
        coin = Cryptocurrency('_'+symbol, db)
        coin.check_db(k_line_type)
        # coin.update_k_lines(k_line_type)
        print("\rSymbol:{:>10s}, Index : {}, K Line : {}, Parameter Searching ".format(index_name, k_line_type, symbol), end='')
        if index_name == "SuperTrend":
            coin.SuperTrend_Parameter(k_line_type,is_save)
        elif index_name == "KDJ":
            coin.KDJ_Parameter(k_line_type,is_save)
    print("\nParameter Search Finish")
        

def press_exit():
    
    print("按任意鍵結束")
    _ = msvcrt.getch()
    sys.exit()
    
def check_backtest_folder():
    
    if not os.path.isdir('./BackTest'):
        os.mkdir('./BackTest')
        
    strategy_list = ['SuperTrend', 'KDJ']
        
    for strategy in strategy_list:
        if not os.path.isdir(f'./BackTest/{strategy}'):
            os.mkdir(f'./BackTest/{strategy}')
        if not os.path.isdir(f'./BackTest/{strategy}/1d'):
            os.mkdir(f'./BackTest/{strategy}/1d')
        if not os.path.isdir(f'./BackTest/{strategy}/12h'):
            os.mkdir(f'./BackTest/{strategy}/12h')
        if not os.path.isdir(f'./BackTest/{strategy}/4h'):
            os.mkdir(f'./BackTest/{strategy}/4h')
        if not os.path.isdir(f'./BackTest/{strategy}/2h'):
            os.mkdir(f'./BackTest/{strategy}/2h')
        if not os.path.isdir(f'./BackTest/{strategy}/1h'):
            os.mkdir(f'./BackTest/{strategy}/1h')
        if not os.path.isdir(f'./BackTest/{strategy}/30m'):
            os.mkdir(f'./BackTest/{strategy}/30m')
    
# def kdj_one(df, ilong = 41, isig = 31):
#     n = df[['Open','High','Low','Close']].to_numpy()
#     c = n[:,3]
    
#     h = np.zeros(n[:,1].shape)
#     for ind, i in enumerate(n[:,1]):
#         h[ind] = np.max(n[ind-ilong+1:ind+1,1]) if ind > ilong else np.max(n[:ind+1,1])
#     l = np.zeros(n[:,2].shape)
#     for ind, i in enumerate(n[:,2]):
#         l[ind] = np.min(n[ind-ilong+1:ind+1,2]) if ind > ilong else np.min(n[:ind+1,2])
    
#     rsv = 100*((c-l)/(h-l))
    
    
#     pk = bcwsma_(rsv, isig)
#     pd = bcwsma_(pk, isig)
    
#     pj = 3*pk-2*pd
    
#     trend = np.ones(pj.shape)
#     buy_singal, sell_singal = 0, 0
#     wallet = Wallet()
#     for ind in range(1,len(pj)):
#         price = n[ind,0]
#         if buy_singal == 1:
#             if wallet.order_list != [] and wallet.order_list[0].type == 'Short':
#                 wallet.order_list[0].close(price)
#                 profit = wallet.order_list[0].profit
#                 wallet.money += profit
#                 history_order = wallet.order_list.pop()
                
#                 if profit >= 0:
#                     wallet.short_win += profit
#                     wallet.short_wcount += 1
#                 else:
#                     wallet.short_lose += profit
#                     wallet.short_lcount += 1
                
#                 wallet.order_history.append(history_order)
            
#             amount = wallet.one_oder_money / price
#             wallet.order_list.append(Order('errortest', amount, price, 'Long'))
#             buy_singal = 0
            
#         elif sell_singal == 1:
#             if wallet.order_list != [] and wallet.order_list[0].type == 'Long':
#                 wallet.order_list[0].close(price)
#                 profit = wallet.order_list[0].profit
#                 wallet.money += profit
#                 history_order = wallet.order_list.pop()
                
#                 if profit >= 0:
#                     wallet.long_win += profit
#                     wallet.long_wcount += 1
#                 else:
#                     wallet.long_lose += profit
#                     wallet.long_lcount += 1
                
#                 wallet.order_history.append(history_order)
                
#             amount = wallet.one_oder_money / price
#             wallet.order_list.append(Order('errortest', amount, price, "Short"))
#             sell_singal = 0
            
#         if pj[ind] >pd[ind]:
#             trend[ind] = 1
#         elif pj[ind] < pd[ind]:
#             trend[ind] = -1
#         else:
#             trend[ind] == trend[ind-1]
#         if trend[ind-1] + trend[ind] == 0:
#             if trend[ind] == 1:
#                 buy_singal = 1
#             else:
#                 sell_singal = 1
    
#     return wallet
    
    
    
def kdj_all(df):
    result = {"Param":[],
              "RTP_LS":[], "RTP_L":[], "RTP_S":[], 
              "ProfitFactor_LS":[], "ProfitFactor_L":[], "ProfitFactor_S":[],
              "Profit_LS":[], "Profit_L":[], "Profit_S":[],
              "WinRate_LS":[], "WinRate_L":[], "WinRate_S":[],
              "WinCount_LS":[], "WinCount_L":[], "WinCount_S":[],
              "Max_win":[], "Max_Profit":[], "Max_Profit_trades":[],"Profit_Date":[],
              "Max_loss":[], "Max_Drawdown":[], "Max_Drawdown_trades":[], "Drawdown_Date":[],
              "Once_Max_Profit":[], "Once_Max_Profit_time":[], "Once_Max_Drawdown":[], "Once_Max_Drawdown_time":[],
              "Average_Long_Profit":[], "Average_Long_Drawdown":[], "Average_Short_Profit":[], "Average_Short_Drawdown":[], "Average_Total_Profit":[], "Average_Total_Drawdown":[]
                  
              }
    wallet_ = []
    
    for ilong in range(1,100):
        wallet_list = np.array([Wallet() for i in range(1,100)])
        n = df[['Open','High','Low','Close']].to_numpy()
        isig = np.array(range(1,100))
        m = 1
        rsv_list = np.zeros(n[:,1].shape)
        trend = np.ones((rsv_list.shape[0],99)).astype(object)
        buy_singal, sell_singal = np.zeros((99,)), np.zeros((99,))
        k_list = np.zeros((rsv_list.shape[0],99)).astype(object)
        d_list = np.zeros((rsv_list.shape[0],99)).astype(object)
        j_list = np.zeros((rsv_list.shape[0],99)).astype(object)
        for ind,row in enumerate(n):
            if ind < ilong-1:
                continue
            c2 = row[3]
            h2 = np.max(n[ind-ilong+1:ind+1,1]) 
            l2 = np.min(n[ind-ilong+1:ind+1,2]) 
            if h2 == l2:
                rsv_list[ind] = rsv_list[ind-1]
            else:
                rsv_list[ind] = 100*((c2-l2)/(h2-l2))
    
            p1 = rsv_list[ind]*m
            p2 = isig-m
    
            p3 = k_list[ind-1,:] 
            k_list[ind,:] = (p1+p2*p3)/isig
    
    
    
            p1 = k_list[ind,:]*m
            p2 = isig-m
    
            p3 = d_list[ind-1,:]
            d_list[ind,:] = (p1+p2*p3)/isig
    
            j_list[ind,:] = 3*k_list[ind] -2*d_list[ind]
            
            
            # if ind < 10371:
            #     continue
            price = row[0]
            
            
            buy_order = np.where(buy_singal == 1)
            for b in buy_order[0]:
                wallet = wallet_list[b]
                if wallet.order_list != [] and wallet.order_list[0].type == 'Short':
                    wallet.order_list[0].close(price)
                    profit = wallet.order_list[0].profit
                    wallet.money += profit
                    history_order = wallet.order_list.pop()
                    
                    if profit >= 0:
                        wallet.short_win += profit
                        wallet.short_wcount += 1
                    else:
                        wallet.short_lose += profit
                        wallet.short_lcount += 1
                    
                    wallet.order_history.append(history_order)
                    wallet.wallet_update()
                amount = wallet.one_oder_money / price
                wallet.order_list.append(Order('errortest', amount, price, "Long", order_date = df['OpenTime'].iloc[ind]))
            sell_order = np.where(sell_singal == 1)
            for s in sell_order[0]:
                wallet = wallet_list[s]
                if wallet.order_list != [] and wallet.order_list[0].type == 'Long':
                    wallet.order_list[0].close(price)
                    profit = wallet.order_list[0].profit
                    wallet.money += profit
                    history_order = wallet.order_list.pop()
                    
                    if profit >= 0:
                        wallet.long_win += profit
                        wallet.long_wcount += 1
                    else:
                        wallet.long_lose += profit
                        wallet.long_lcount += 1
                    
                    wallet.order_history.append(history_order)
                    wallet.wallet_update()
                amount = wallet.one_oder_money / price
                wallet.order_list.append(Order('errortest', amount, price, "Short", order_date = df['OpenTime'].iloc[ind]))
            
            buy_singal, sell_singal = np.zeros((99,)), np.zeros((99,))
            
            
            
            long_x = np.where(j_list[ind,:] > d_list[ind,:])
            short_x = np.where(j_list[ind,:] < d_list[ind,:])
            no_x = np.where(j_list[ind,:] == d_list[ind,:])
            last_t = np.array(trend[ind-1, no_x])
            trend[ind,long_x], trend[ind,short_x], trend[ind,no_x] = 1, -1, last_t
            
            buy_ = np.where(((trend[ind-1,:]+trend[ind,:]) == 0 )& (trend[ind,:]==1))
            sell_ = np.where(((trend[ind-1,:]+trend[ind,:]) == 0 )& (trend[ind,:]==-1))
            buy_singal[buy_], sell_singal[sell_] = 1, 1
            
            
        
        for ind, w in enumerate(wallet_list):
            
            result["Param"].append('ilong:{:02}, isig:{:02}'.format(ilong,(ind+1)))
            
            
            result["ProfitFactor_LS"].append((w.long_win + w.short_win)/-1*(w.long_lose + w.short_lose) if (w.long_lose + w.short_lose) != 0 else (w.long_win + w.short_win))
            result["ProfitFactor_L"].append(w.long_win / -1*w.long_lose if w.long_lose != 0 else w.long_win)
            result["ProfitFactor_S"].append(w.short_win/ -1*w.short_lose if w.short_lose != 0 else w.short_win)
            ls,l,s = w.long_win + w.short_win + w.long_lose + w.short_lose, w.long_win + w.long_lose, w.short_win +w.short_lose
            lsw,lw,sw = [w.long_wcount+w.short_wcount, w.long_wcount+w.short_wcount+w.long_lcount+w.short_lcount], [w.long_wcount, w.long_wcount+w.long_lcount], [w.short_wcount, w.short_wcount+w.short_lcount]
    
    
            result["Profit_LS"].append(ls)
            result["Profit_L"].append(l)
            result["Profit_S"].append(s)
            result["RTP_LS"].append(ls/w.one_oder_money)
            result["RTP_L"].append(l/w.one_oder_money)
            result["RTP_S"].append(s/w.one_oder_money)
            result["WinRate_LS"].append(lsw[0]/lsw[1] if lsw[1] != 0 else lsw[0])
            result["WinRate_L"].append(lw[0]/lw[1] if lw[1] != 0 else lw[0])
            result["WinRate_S"].append(sw[0]/sw[1] if sw[1] != 0 else sw[0])
            result["WinCount_LS"].append(lsw)
            result["WinCount_L"].append(lw)
            result["WinCount_S"].append(sw)
            
            result["Max_win"].append(w.max_win)
            result["Max_Profit"].append(w.max_profit)
            result["Max_Profit_trades"].append(w.max_profit_trades)
            result["Profit_Date"].append(w.profit_date)
            
            result["Max_loss"].append(w.max_loss)
            result["Max_Drawdown"].append(w.max_drawdown)
            result["Max_Drawdown_trades"].append(w.max_drawdown_trades)
            result["Drawdown_Date"].append(w.drawdown_date)
                    
            result['Once_Max_Profit'].append(w.once_max_profit*100)
            result['Once_Max_Profit_time'].append(w.once_max_profit_time)
            result['Once_Max_Drawdown'].append(w.once_max_drawdown*100)
            result['Once_Max_Drawdown_time'].append(w.once_max_drawdown_time)
            
            result['Average_Long_Profit'].append(w.average_long_profit*100)
            result['Average_Long_Drawdown'].append(w.average_long_drawdown*100)
            result['Average_Short_Profit'].append(w.average_short_profit*100)
            result['Average_Short_Drawdown'].append(w.average_short_drawdown*100)
            result['Average_Total_Profit'].append(w.average_profit*100)
            result['Average_Total_Drawdown'].append(w.average_drawdown*100)
            
            
            
        wallet_.append(wallet_list)
    strategy_reback_result = pd.DataFrame(result)
    
    return strategy_reback_result, wallet_
    
# def bcwsma_(array, isig, m = 1):
#     bcwsma = np.zeros(array.shape)
#     p1 = array*m
#     p2 = isig-m
    
#     for ind in range(len(bcwsma)):
#         p3 = bcwsma[ind-1] if ind > 0 else 0
#         bcwsma[ind] = (p1[ind]+p2*p3)/isig
        
#     return bcwsma
    
    
    
if __name__ == "__main__":
    a = 1
    # db = DBsystem("crytodb")
    # btc = Cryptocurrency("_BTCUSDT", db)
    # btc.check_db('1d')
    # df = btc.SuperTrend('1d','hl2',2,7.3)
    # df_super = btc.SuperTrend_Parameter('1d',0)
    # df_super_new = btc.SuperTrend_Parameter_new('1d',0)
    # '''
    #%%
    
    try:
        print("Welcome Index Parameter Search System....")
        # client = connect_binance()
        client = Client('','')
        
        db = DBsystem("crytodb")
        
        check_backtest_folder()
        
        
        btc = Cryptocurrency("_BTCUSDT", db)
        btc.get_db_last_date()
        
        
        all_ticker = client.get_exchange_info()['symbols']
        cyptor_list = []
        for i in all_ticker:
            if i['quoteAsset'] == 'USDT' and 'LEVERAGED' not in i['permissions'] and i['status'] == 'TRADING' :
                cyptor_list.append(i['symbol'])
                
        
        system_options = '0'
        
        while system_options != 'e':
            if system_options == '0':
                system_options = input("選擇目的 (1)更新K線資料 (2)指標參數搜尋 (e)離開程式 : ")
            elif system_options == '1':
                update_k_lines_db(cyptor_list, db)
                print("\nK線更新完畢.... \n")
                system_options = '0'
            elif system_options == '2':
                system_options = '0'
                while True:
                    index_numel = input("選擇指標 (1) SuperTrend (2) KDJ (0)回主選單 (e)離開程式 : ")
                    if index_numel == '1':
                        index_name = 'SuperTrend'
                        system_options = 'choice kline'
                        break
                    elif index_numel == '2':
                        index_name = 'KDJ'
                        system_options = 'choice kline'
                        break
                    elif index_numel == '0':
                        break
                    elif index_numel == 'e':
                        system_options = 'e'
                        break
                    else:
                        print('\n輸入錯誤請重新選擇 \n')
            elif system_options == 'choice kline':
                system_options = '0'
                while True:
                    k_type = input("請選擇K線 (1)1d (2)12h (3)4h (4)2h (5)1h (6)30m (0)回主選單 (e)離開程式 : ")
                    if k_type == '1':
                        k_line_type = '1d'
                        system_options = 'calcul index'
                    elif k_type == '2':
                        k_line_type = '12h'
                        system_options = 'calcul index'
                    elif k_type == '3':
                        k_line_type = '4h'
                        system_options = 'calcul index'
                    elif k_type == '4':
                        k_line_type = '2h'
                        system_options = 'calcul index'
                    elif k_type == '5':
                        k_line_type = '1h'
                        system_options = 'calcul index'
                    elif k_type == '6':
                        k_line_type = '30m'
                        system_options = 'calcul index'
                    elif k_type == '0':
                        break
                    elif k_type == 'e':
                        system_options = 'e'
                    else:
                        print("\nk線選擇錯誤 \n")
                        continue
                    break
            #     if index_numel
            elif system_options == 'calcul index':
                system_options = '0'
                print('\n指數 {}, K線 {}\n'.format(index_name, k_line_type))
                symbol = input("請輸入想要計算的幣別並依','當間隔，若想全測則輸入all 回主選單請輸入(0) : \n")
                if symbol == '0':
                    system_options = '0'
                    continue
                if symbol.replace(' ','').lower() == 'all':
                    symbol_list = cyptor_list
                elif symbol.replace(' ','') == '':
                    system_options = 'calcul index'
                    symbol_list = '幣別不得為空'
                    continue
                else:
                    symbol_list = [i+'USDT' for i in  symbol.replace(' ','').upper().replace('USDT','').split(',')]
                is_save = int(input("是否輸出結果? (1) Yes (0) No : "))
                if is_save not in [0,1]:
                    is_save = 1
                parameter_search(symbol_list, index_name, k_line_type, is_save, db)
                # print(symbol_list, index_name, k_line_type, is_save)
                
            else:
                system_options = '0'
    except Exception as e:
        print(f"Error : {e}")
    # '''
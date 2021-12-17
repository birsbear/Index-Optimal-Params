# 專案簡介

對接幣安API，抓取所有USDT對的30m/1h/2h/4h/12h/1d K線，存入資料庫中，實現不同的指標並回測一定範圍內所有可變動參數，輸出其結果進行最佳參數評估

目前實現並實裝指標為: KDJ、SuperTrend

CLI版本已完成，GUI版本新增移動止損回測正在開發中

# 專案適用對象

此程式適用於想要尋找最佳指標參數的使用者

# 專案功能需求

## CLI版本

- DataBase
    - DataBase 建立
    - DataBase 存讀取
- Binance API
    - 爬取各USDT交易對
    - 爬取交易對k線資料
- CLI
    - 判斷最後一次更新k線資料庫時間
    - 更新k線資料庫
    - 選擇回測指標
    - 選擇回測k線
    - 選擇回測幣種
    - 是否輸出結果
- Strategy
    - KDJ
    - SuperTrend
- Backtesting
    - 回測項目：
        - 多/空/多空 : 獲利因子, 獲利, 回報率, 勝率, 勝利次數
        - 最大: 獲利,連續獲利,連續獲利次數,虧損,連續虧損,連續虧損次數
        - 單次最大:獲利,虧損
        - 平均多/空/多空:獲利,虧損

## GUI版本

- 主回測參數設定
    - 輸入幣種 (input text )
        - all : 幣安全幣種USDT對，按上幣安時間排序
        - [Symbol1, Symbol2, Symbol3, Symbol4] : 幣的代號，大小寫皆可，皆為USDT對(不用輸入USDT)
    - 選擇回測範圍(日曆)
    - 選擇線圖(下拉式 : 1D 12H 4H 2H 1H 30m)
    - 選擇回測策略(下拉式 : SuperTrend、KDJ)
        - 選取不同策略時，會切換不同的參數設定畫面
    - 選擇回測方式(下拉式 : 每次固定金額、每次固定數量、每次固定本金比例)
        - 選取不同回測方式時，會切換不同的回測設定畫面
    - 選擇策略參數(Check Box 移動止損)
        - 會根據選擇策略不同切換內容
        - 移動止損初階段沒有觸發價
- 回測設定
    - 固定金額
        - 設定初始本金
        - 每次入場金額
    - 固定數量
        - 設定初始本金
        - 每次入場數量
    - 固定比例
        - 設定初始本金
        - 每次入場比例
- 策略參數
    - SuperTrend
        - 選擇搜尋目標(Select Box 全搜索、指定參數)
            - 參數欄位 :
                - Source : close, hl2, hlc3, ohlc4
                - Period :  range(1,8)
                - Factory : arange(0.1, 10.1, 0.1)
            - 全搜索 : 反灰參數欄位
            - 指定參數 : 開放參數欄位
    - KDJ
        - 選擇搜尋目標(Select Box 全搜索、指定參數)
            - 參數欄位 :
                - Long:  range(1,101)
                - Sig: arange(1,101)
            - 全搜索 : 反灰參數欄位
            - 指定參數 : 開放參數欄位
- 移動止損
    - 設定止損比例
- 資料庫
    - 建立幣價資料庫
    - 建立指標資料庫（日後不重複計算
    - 判斷資料庫是否最新資料

# 功能架構圖
![Functional architecture diagram](https://github.com/birsbear/Index-Optimal-Params/blob/main/Index-Optimal-Params.png)

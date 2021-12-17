# Index-Optimal-Params-
# 專案簡介

對接幣安API，抓取所有USDT對的30m/1h/2h/4h/12h/1d K線，存入資料庫中，實現不同的指標並回測一定範圍內所有可變動參數，輸出其結果進行最佳參數評估

目前實現並實裝指標為: KDJ、SuperTrend

CLI版本已完成，GUI版本新增移動止損回測正在開發中

---

測試用DB有 BTC ETH的資料，可以直接使用

`python Index_Parameter_Search_System.py`

選擇幣種時可選擇輸入 BTC ETH

![demo](https://github.com/birsbear/Index-Optimal-Params/blob/main/SuperTrend.png)

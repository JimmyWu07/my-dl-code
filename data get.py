import pandas_datareader.data as web
import datetime



start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2024, 12, 31)

# 雅虎财经代码：沪深300是000300.SS
df = web.DataReader('000300.SS', 'yahoo', start, end)
df = df.reset_index()
df = df.rename(columns={'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
df.to_csv('hs300_real.csv', index=False, encoding='utf-8-sig')
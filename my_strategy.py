"""
基于均线与风险控制的量化策略研究
内容:
1. 数据获取
2. 数据清洗
3. 特征构建
4. 策略逻辑
5. 回撤
6. 风险指标
7. 极端行情测试
8. 可视化
9. 策略分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df['MA5'] = df['close'].rolling(5).mean()
df['MA10'] = df['close'].rolling(10).mean()
df['MA20'] = df['close'].rolling(20).mean()
df = df.dropna(subset=['MA5', 'MA10', 'MA20'])

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
# 1. 生成金叉（买入）、死叉（卖出）信号
# 金叉：MA5上穿MA20（前一天MA5<=MA20，当天MA5>MA20）
df['golden_cross'] = (df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1))
# 死叉：MA5下穿MA20（前一天MA5>=MA20，当天MA5<MA20）
df['death_cross'] = (df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1))

# 2. 整合信号（1=买入，-1=卖出，0=无操作）
df['trade_signal'] = 0
df.loc[df['golden_cross'], 'trade_signal'] = 1
df.loc[df['death_cross'], 'trade_signal'] = -1

# 连续2天MA5 > MA20，才确认金叉
df['valid_golden_cross'] = (
    (df['MA5'] > df['MA20']) & 
    (df['MA5'].shift(1) > df['MA20'].shift(1)) &
    (df['MA5'].shift(2) <= df['MA20'].shift(2))
)
# 金叉时成交量 > 5日均量的1.2倍
df['vol_ma5'] = df['volume'].rolling(5).mean()
df['valid_golden_cross'] = (
    (df['MA5'] > df['MA20']) & 
    (df['MA5'].shift(1) <= df['MA20'].shift(1)) &
    (df['volume'] > 1.2 * df['vol_ma5'])
)
df['distance_to_ma20'] = abs(df['close'] - df['MA20']) / df['MA20']
df['valid_golden_cross'] = (
    (df['MA5'] > df['MA20']) & 
    (df['MA5'].shift(1) <= df['MA20'].shift(1)) &
    (df['distance_to_ma20'] > 0.01)  # 离均线1%以上
)
# -------------------------- 回测部分 --------------------------
df['return'] = df['close'].pct_change() #计算收益率，市场收益；
df['signal'] = 0
df.loc[df['close'] > df['MA20'], 'signal'] = 1 #做多 即五日线大于二十日线；
df['strategy_return'] = df['signal'].shift(1) * df['return'] #计算策略收益 shift(1):防止未来函数
df['cumulative_return'] = (1 + df['return']).cumprod() - 1 #一直持有收益 #cumprod：累乘
df['cumulative_strategy_return'] = (1 + df['strategy_return']).cumprod() - 1 #按照策略持有收益

# -------------------------- PCA部分 --------------------------
#PCA部分适合拿来降噪，提取关键特征，可视化部分可以忽略
features = ['open', 'high', 'low', 'close', 'volume'] #提取特征
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 把所有特征缩到同一个量级
# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0] #价格 + 成交量的整体趋势和波动
df['pca2'] = X_pca[:, 1] #价格和成交量的背离、市场的异常波动

# --------------------------夏普比率部分 --------------------------
mean_return = df['strategy_return'].mean() #计算收益均值
std_return = df['strategy_return'].std() #计算收益标准差(波动率)
sharpe_ratio = mean_return / std_return
sharpe_ratio = (
    mean_return / std_return
) * np.sqrt(50)
print("夏普比率:", sharpe_ratio)
if sharpe_ratio <= 1:
    print("策略水平一般")
elif (sharpe_ratio > 1) & (sharpe_ratio <=2):
    print("策略水平还不错")
elif (sharpe_ratio > 2) & (sharpe_ratio <=3):
    print("策略水平很强")
else:
    print("这个策略很少见")


# ------------------- 可视化部分 -------------------
plt.figure(figsize=(12, 6))  # 设置画布大小

# 画收盘价折线 五日十日二十日均线
plt.plot(df.index, df['close'], label='close', color='blue', marker='o')
plt.plot(df.index, df['MA5'], label='MA5', color='red', linestyle='--', linewidth=1.5)
plt.plot(df.index, df['MA10'], label='MA10', color='orange', linestyle='-.', linewidth=1.5)
plt.plot(df.index, df['MA20'], label='MA20', color='green', linestyle=':', linewidth=1.5)


# 美化图表
plt.title('Stock closing price and five-day moving average', fontsize=14)
plt.xlabel('date', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.xticks(rotation=45)  # 日期标签旋转45度，避免重叠
plt.tight_layout()  # 自动调整布局，防止标签被截断
plt.show()

# 回测收益可视化
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['cumulative_return'], 
         label='benchmark holding income', color='gray', linewidth=2)#基准持有收益
plt.plot(df.index, df['cumulative_strategy_return'], 
         label='EMA strategy return', color='red', linewidth=2)#均线策略收益

plt.title('Strategy Backtest: Cumulative Return Comparison')#策略回测：累计收益比
plt.xlabel('date')
plt.ylabel('cumulative rate of return')#累计收益率
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# ------------------- PCA 特征可视化 -------------------
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['pca1'], label='PCA first part', color='blue', linewidth=2)
plt.plot(df.index, df['pca2'], label='PCA second part', color='red', linewidth=2)

plt.title('PCA The two core features extracted')#PCA 提取的两个核心特征
plt.xlabel('date')
plt.ylabel('pca_number')#pca数值
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='close', color='blue', marker='.')
plt.plot(df.index, df['MA5'], label='MA5', color='red', linestyle='--', linewidth=1.5)
plt.plot(df.index, df['MA10'], label='MA10', color='orange', linestyle='-.', linewidth=1.5)
plt.plot(df.index, df['MA20'], label='MA20', color='green', linestyle=':', linewidth=1.5)

# 新增：标记买入/卖出信号点
buy_points = df[df['trade_signal'] == 1]
sell_points = df[df['trade_signal'] == -1]
plt.scatter(buy_points.index, buy_points['close'], 
            color='green', marker='^', s=100, label='Buy', zorder=5)  # 绿色上箭头，放大显示
plt.scatter(sell_points.index, sell_points['close'], 
            color='red', marker='v', s=100, label='Sell', zorder=5)   # 红色下箭头，放大显示


plt.title('Stock Price + MA + Trade Signals', fontsize=14)
plt.xlabel('date', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

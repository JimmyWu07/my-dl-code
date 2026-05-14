import numpy as np
import pandas as pd

class DoubleMAStrategy:
    """
    双均线策略封装类
    """
    def __init__(self, df, short_window=5, long_window=20):
        self.df = df.copy()
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        # 1. 计算均线
        self.df['MA_short'] = self.df['close'].rolling(window=self.short_window).mean()
        self.df['MA_long'] = self.df['close'].rolling(window=self.long_window).mean()

        # 核心修改：删掉所有含有 NaN 的行（即删掉预热期）
        # 这样你的 df 就会从第 20 天（或你设置的最长均线那天）正式开始
        self.df = self.df.dropna().copy()

        # 2. 生成信号 (在没有空值的数据集上进行计算)
        self.df['signal'] = 0
        self.df.loc[self.df['MA_short'] > self.df['MA_long'], 'signal'] = 1
        self.df.loc[self.df['MA_short'] < self.df['MA_long'], 'signal'] = -1
        
        return self.df
        

    def calculate_returns(self):
        # 3. 计算收益率
        self.df['market_return'] = self.df['close'].pct_change()
        self.df['strategy_return'] = self.df['signal'].shift(1) * self.df['market_return']
        
        # 4. 计算累计收益
        self.df['cum_market_return'] = (1 + self.df['market_return']).cumprod()
        self.df['cum_strategy_return'] = (1 + self.df['strategy_return']).cumprod()
        
        return self.df

    def get_performance(self):
        # 5. 计算夏普比率 (年化)
        returns = self.df['strategy_return'].dropna()
        if len(returns) > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        return sharpe
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

        self.df = self.df.dropna().copy()

        # 2. 生成信号 (在没有空值的数据集上进行计算)
        self.df['signal'] = 0
        self.df.loc[self.df['MA_short'] > self.df['MA_long'], 'signal'] = 1
        self.df.loc[self.df['MA_short'] < self.df['MA_long'], 'signal'] = -1

        # 进阶逻辑：只有今天交叉了才算信号
        # 今天金叉 且 昨天不是金叉
        self.df['position'] = (self.df['MA_short'] > self.df['MA_long']).astype(int)
        self.df['signal'] = self.df['position'].diff() 
        # 此时：1 代表买入，-1 代表卖出，0 代表持有或不操作
        
        return self.df
        

    def calculate_returns(self):
        # 3. 计算收益率
        self.df['market_return'] = self.df['close'].pct_change()
        self.df['strategy_return'] = self.df['signal'].shift(1) * self.df['market_return']
        
        # 4. 计算累计收益
        self.df['cum_market_return'] = (1 + self.df['market_return']).cumprod()
        self.df['cum_strategy_return'] = (1 + self.df['strategy_return']).cumprod()
        # 5. 添加手续费 后期再加滑点
        commission = 0.001  # 0.1% 手续费
        self.df['strategy_return'] = self.df['signal'].shift(1) * self.df['market_return'] - commission * (self.df['signal'] != self.df['signal'].shift(1)).astype(int)
        
        return self.df

    def get_performance(self):
        # 6. 计算夏普比率 (年化)
        returns = self.df['strategy_return'].dropna()
        if len(returns) > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        return sharpe
    
    def get_max_drawdown(self):
        # 7. 计算策略收益曲线的最大回撤
    
        cum_return = self.df['cum_strategy_return']
        running_max = cum_return.expanding().max()
        drawdown = (cum_return - running_max) / running_max
        max_drawdown = drawdown.min()
        return max_drawdown
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

        # 2. 删除NaN
        self.df = self.df.dropna().copy()

        # 3. 生成交易信号
        self.df['signal'] = 0
        # 金叉：今天 MA_short > MA_long，昨天 MA_short <= MA_long
        self.df.loc[(self.df['MA_short'] > self.df['MA_long']) & 
                    (self.df['MA_short'].shift(1) <= self.df['MA_long'].shift(1)), 'signal'] = 1
        # 死叉：今天 MA_short < MA_long，昨天 MA_short >= MA_long
        self.df.loc[(self.df['MA_short'] < self.df['MA_long']) & 
                    (self.df['MA_short'].shift(1) >= self.df['MA_long'].shift(1)), 'signal'] = -1
    
        # 4. 生成仓位列
        self.df['position'] = 0
        position = 0
        for i in range(len(self.df)):
            if self.df['signal'].iloc[i] == 1:      # 买入信号
                position = 1
            elif self.df['signal'].iloc[i] == -1:   # 卖出信号
                position = 0
            self.df.loc[self.df.index[i], 'position'] = position
        return self.df

    def calculate_returns(self):
        self.df['market_return'] = self.df['close'].pct_change()
        # 用 position 计算连续持仓收益
        self.df['strategy_return'] = self.df['position'].shift(1) * self.df['market_return']
        
        self.df['cum_market_return'] = (1 + self.df['market_return']).cumprod()
        self.df['cum_strategy_return'] = (1 + self.df['strategy_return']).cumprod()
        commission = 0.0003
        self.df['strategy_return'] = self.df['position'].shift(1) * self.df['market_return'] - self.df['signal'].abs() * commission
        
        return self.df
    def calculate_adx(self, period=14):
        # adx25%过滤器
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        # 计算 +DM 和 -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # 计算真实波幅 TR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 平滑
        atr = tr.rolling(period).mean()
        smooth_plus_dm = plus_dm.rolling(period).mean()
        smooth_minus_dm = minus_dm.rolling(period).mean()
        
        # 计算 +DI 和 -DI
        plus_di = 100 * (smooth_plus_dm / atr)
        minus_di = 100 * (smooth_minus_dm / atr)
        
        # 计算 DX 和 ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    def get_performance(self):
        # 6. 计算夏普比率 (年化)
        returns = self.df['strategy_return'].dropna()
        if len(returns) > 0 and returns.std() > 0:
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
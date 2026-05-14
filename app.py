import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from config import DATA_DIR, STRATEGY_PARAMS
from strategy import DoubleMAStrategy

# 设置页面标题
st.set_page_config(page_title="量化运维可视化平台", layout="wide")

st.title("📈 量化策略交互回测平台")
st.sidebar.markdown(f"**运行人:241733402_吴鸿鸣**")

# --- 侧边栏：参数调节 ---
st.sidebar.header("策略参数设置")
target_file = st.sidebar.selectbox("选择数据集", ["normal_market.csv", "extreme_market.csv"])
short_val = st.sidebar.slider("短期均线窗口", 2, 30, STRATEGY_PARAMS['short_window'])
long_val = st.sidebar.slider("长期均线窗口", 10, 100, STRATEGY_PARAMS['long_window'])

# --- 数据读取 ---
csv_path = os.path.join(DATA_DIR, target_file)

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, encoding=STRATEGY_PARAMS['encoding'])
    
    # 实例化策略
    algo = DoubleMAStrategy(df, short_window=short_val, long_window=long_val)
    df_res = algo.generate_signals()
    df_res = algo.calculate_returns()
    
    # --- 布局：上方显示核心指标 ---
    col1, col2, col3 = st.columns(3)
    col1.metric("最终累计收益", f"{df_res['cum_strategy_return'].iloc[-1]:.2f}x")
    col2.metric("年化夏普比率", f"{algo.get_performance():.2f}")
    col3.metric("测试天数", len(df_res))

    # --- 图表 1：双均线回测图 ---
    st.subheader("双均线交易信号可视化")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_res['date'], df_res['close'], label='Close Price', alpha=0.5)
    ax1.plot(df_res['date'], df_res['MA_short'], label=f'MA {short_val}')
    ax1.plot(df_res['date'], df_res['MA_long'], label=f'MA {long_val}')
    
    # 标记买卖点
    buy_sig = df_res[df_res['signal'] == 1]
    sell_sig = df_res[df_res['signal'] == -1]
    ax1.scatter(buy_sig['date'], buy_sig['close'], marker='^', color='red', label='Buy', s=100)
    ax1.scatter(sell_sig['date'], sell_sig['close'], marker='v', color='green', label='Sell', s=100)
    
    plt.xticks(df_res['date'][::20], rotation=45)
    plt.legend()
    st.pyplot(fig1)

    # --- 图表 2：累计收益对比 ---
    st.subheader("策略收益 vs 市场基准")
    st.line_chart(df_res.set_index('date')[['cum_market_return', 'cum_strategy_return']])

    # --- 数据表格预览 ---
    if st.checkbox("查看原始数据明细"):
        st.write(df_res.tail(20))

else:
    st.error(f"找不到文件: {csv_path}")
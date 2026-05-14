import os
import sys
import pandas as pd

# 确保 Python 能找到同目录下的 config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import DATA_DIR, STRATEGY_PARAMS
from strategy import DoubleMAStrategy

def main():
    # 1. 设定文件名
    file_name = "extreme_market.csv" 
    csv_path = os.path.join(DATA_DIR, file_name)
    
    # 打印一下实际搜索的路径，方便你核对
    print(f"🔍 正在读取文件: {csv_path}")
    
    # 2. 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件！请确认 {file_name} 是否直接放在 py 文件夹里。")
        return

    # 3. 读取并运行
    try:
        df = pd.read_csv(csv_path, encoding=STRATEGY_PARAMS['encoding'])
        
        algo = DoubleMAStrategy(
            df, 
            short_window=STRATEGY_PARAMS['short_window'], 
            long_window=STRATEGY_PARAMS['long_window']
        )
        
        df_result = algo.generate_signals()
        df_result = algo.calculate_returns()
        
        print("="*40)
        print(f"量化运维监控 - 运行人: 241733402_吴鸿鸣")
        print("-" * 40)
        print(f"策略表现：年化夏普比率 = {algo.get_performance():.2f}")
        print("="*40)
        
    except Exception as e:
        print(f"⚠️ 运行出错: {e}")

if __name__ == "__main__":
    main()
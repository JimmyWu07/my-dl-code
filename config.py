import os

# 获取当前 config.py 所在的文件夹路径 (也就是你的 py 文件夹)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 关键修改：直接使用 BASE_DIR，不再拼接 "data" 文件夹
DATA_DIR = BASE_DIR 

STRATEGY_PARAMS = {
    'short_window': 5,
    'long_window': 20,
    'encoding': 'utf-8-sig'
}
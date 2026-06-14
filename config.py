import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = BASE_DIR

STRATEGY_PARAMS = {
    'short_window': 5,
    'long_window': 20,
    'adx_period': 14,
    'adx_threshold': 12,
    'commission': 0.0003,
    'encoding': 'utf-8-sig'
}
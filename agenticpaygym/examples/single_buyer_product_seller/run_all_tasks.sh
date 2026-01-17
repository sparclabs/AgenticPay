#!/bin/bash

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 运行所有任务
python Task1_basic_price_negotiation.py
python Task2_close_price_negotiation.py
python Task3_close_to_market_price_negotiation.py

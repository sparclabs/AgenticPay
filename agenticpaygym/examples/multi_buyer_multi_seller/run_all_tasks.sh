#!/bin/bash

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 运行所有任务
python Task1_parallel_two_buyer_two_seller_negotiation.py
python Task2_parallel_three_buyer_three_seller_negotiation.py
python Task3_sequential_two_buyer_two_seller_negotiation.py
python Task4_sequential_three_buyer_three_seller_negotiation.py

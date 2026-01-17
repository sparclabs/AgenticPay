#!/bin/bash

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 运行所有任务
python Task1_multi_product_negotiation.py
python Task2_two_product_negotiation.py
python Task3_five_product_negotiation.py
python Task4_select_three_from_five_negotiation.py

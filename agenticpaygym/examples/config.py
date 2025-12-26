"""Configuration file for AgenticPayGym examples

This file contains common configuration variables used across different
negotiation examples, including reward weights, aggregation methods, and
environment parameters.
"""

# Reward weights configuration
# These weights control the relative importance of different reward components
reward_weights = {
    "buyer_savings": 1.0,      # 买方节省权重
    "seller_profit": 1.0,      # 卖方利润权重
    "time_cost": 0.1,          # 时间成本权重（降低影响）
}

# Reward aggregation methods
# Options: "average", "max", "min"
buyer_reward_aggregation = "average"  # 买方奖励聚合方法
seller_reward_aggregation = "average"  # 卖方奖励聚合方法

# Environment parameters
max_rounds = 20  # 最大谈判轮数
price_tolerance = 0.0  # 价格容忍度（用于判断价格是否匹配）


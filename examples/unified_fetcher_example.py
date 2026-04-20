"""
统一数据获取接口使用示例
"""

from lib.unified_fetcher import fetch_stock_data_unified

# 示例1: 获取美股数据
print("=== 示例1: 获取美股数据 ===")
us_data = fetch_stock_data_unified("AAPL", "US", days=30)
print(f"数据形状: {us_data.shape}")
print(f"列名: {us_data.columns.tolist()}")
print(f"最新收盘价: {us_data['Close'].iloc[-1]}")

# 示例2: 获取A股数据
print("\n=== 示例2: 获取A股数据 ===")
cn_data = fetch_stock_data_unified("000001.SZ", "CN", days=30)
print(f"数据形状: {cn_data.shape}")
print(f"列名: {cn_data.columns.tolist()}")
print(f"最新收盘价: {cn_data['Close'].iloc[-1]}")

# 示例3: 获取港股数据
print("\n=== 示例3: 获取港股数据 ===")
hk_data = fetch_stock_data_unified("00700.HK", "HK", days=30)
print(f"数据形状: {hk_data.shape}")
print(f"列名: {hk_data.columns.tolist()}")
print(f"最新收盘价: {hk_data['Close'].iloc[-1]}")

# 示例4: 使用缓存（第二次获取相同数据会从缓存读取）
print("\n=== 示例4: 使用缓存 ===")
us_data_cached = fetch_stock_data_unified("AAPL", "US", days=30)
print(f"从缓存获取数据: {us_data_cached.shape}")

# 示例5: 错误处理（不支持的市场）
print("\n=== 示例5: 错误处理 ===")
invalid_data = fetch_stock_data_unified("AAPL", "XX", days=30)
print(f"不支持的市场返回空DataFrame: {invalid_data.empty}")

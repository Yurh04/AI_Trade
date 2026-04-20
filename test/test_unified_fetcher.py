"""
测试统一数据获取接口
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.unified_fetcher import fetch_stock_data_unified

print("测试统一数据获取接口...")

# 测试美股（使用模拟数据，避免网络请求）
print("\n=== 测试美股 (AAPL) ===")
try:
    df = fetch_stock_data_unified("AAPL", "US", days=5)
    if not df.empty:
        print(f"✅ 数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前3行:\n{df.head(3)}")
    else:
        print("❌ 获取失败")
except Exception as e:
    print(f"❌ 错误: {e}")

# 测试A股（使用模拟数据）
print("\n=== 测试A股 (000001.SZ) ===")
try:
    df = fetch_stock_data_unified("000001.SZ", "CN", days=5)
    if not df.empty:
        print(f"✅ 数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前3行:\n{df.head(3)}")
    else:
        print("❌ 获取失败")
except Exception as e:
    print(f"❌ 错误: {e}")

# 测试港股（使用模拟数据）
print("\n=== 测试港股 (00700.HK) ===")
try:
    df = fetch_stock_data_unified("00700.HK", "HK", days=5)
    if not df.empty:
        print(f"✅ 数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前3行:\n{df.head(3)}")
    else:
        print("❌ 获取失败")
except Exception as e:
    print(f"❌ 错误: {e}")

# 测试缓存
print("\n=== 测试缓存功能 ===")
print("再次获取AAPL数据（应该从缓存读取）...")
try:
    df = fetch_stock_data_unified("AAPL", "US", days=5)
    if not df.empty:
        print(f"✅ 从缓存获取数据: {df.shape}")
except Exception as e:
    print(f"❌ 错误: {e}")

print("\n✅ 测试完成")

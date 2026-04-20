"""
测试港股数据获取功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.data_fetcher import fetch_hk_stock_data, generate_simulated_hk_data

print("=" * 60)
print("测试港股数据获取功能")
print("=" * 60)

# 测试腾讯控股（00700.HK）
print("\n=== 测试 00700.HK（腾讯控股）===")
df = fetch_hk_stock_data("00700.HK", days=30)
if not df.empty:
    print(f"✅ 数据形状: {df.shape}")
    print(f"✅ 列名: {df.columns.tolist()}")
    print(f"✅ 前5行:\n{df.head()}")
    print(f"✅ 数据类型:\n{df.dtypes}")
else:
    print("⚠️ 获取失败，使用模拟数据")
    df = generate_simulated_hk_data("00700.HK", days=30)
    print(f"✅ 模拟数据形状: {df.shape}")
    print(f"✅ 前5行:\n{df.head()}")

# 测试中国移动（00941.HK）
print("\n=== 测试 00941.HK（中国移动）===")
df = fetch_hk_stock_data("00941.HK", days=30)
if not df.empty:
    print(f"✅ 数据形状: {df.shape}")
    print(f"✅ 列名: {df.columns.tolist()}")
    print(f"✅ 前5行:\n{df.head()}")
else:
    print("⚠️ 获取失败，使用模拟数据")
    df = generate_simulated_hk_data("00941.HK", days=30)
    print(f"✅ 模拟数据形状: {df.shape}")
    print(f"✅ 前5行:\n{df.head()}")

# 测试模拟数据生成
print("\n=== 测试模拟数据生成 ===")
df = generate_simulated_hk_data("00700.HK", days=10)
print(f"✅ 模拟数据形状: {df.shape}")
print(f"✅ 列名: {df.columns.tolist()}")
print(f"✅ 前5行:\n{df.head()}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

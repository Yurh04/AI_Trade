"""
测试港股模拟数据生成功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.data_fetcher import generate_simulated_hk_data

print("=" * 60)
print("测试港股模拟数据生成功能")
print("=" * 60)

# 测试腾讯控股（00700.HK）
print("\n=== 测试 00700.HK（腾讯控股）模拟数据 ===")
df = generate_simulated_hk_data("00700.HK", days=10)
print(f"✅ 模拟数据形状: {df.shape}")
print(f"✅ 列名: {df.columns.tolist()}")
print(f"✅ 前5行:\n{df.head()}")
print(f"✅ 数据类型:\n{df.dtypes}")

# 测试中国移动（00941.HK）
print("\n=== 测试 00941.HK（中国移动）模拟数据 ===")
df = generate_simulated_hk_data("00941.HK", days=10)
print(f"✅ 模拟数据形状: {df.shape}")
print(f"✅ 前5行:\n{df.head()}")

# 测试未知股票代码
print("\n=== 测试未知股票代码（09999.HK）模拟数据 ===")
df = generate_simulated_hk_data("09999.HK", days=10)
print(f"✅ 模拟数据形状: {df.shape}")
print(f"✅ 前5行:\n{df.head()}")

print("\n" + "=" * 60)
print("模拟数据测试完成")
print("=" * 60)

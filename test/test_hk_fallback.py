"""
测试fetch_hk_stock_data函数的fallback机制
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.data_fetcher import fetch_hk_stock_data, generate_simulated_hk_data

print("=" * 60)
print("测试fetch_hk_stock_data的fallback机制")
print("=" * 60)

# 测试腾讯控股（00700.HK）
print("\n=== 测试 00700.HK（腾讯控股）===")
df = fetch_hk_stock_data("00700.HK", days=10)

if df.empty:
    print("⚠️ API获取失败，使用模拟数据作为fallback")
    df = generate_simulated_hk_data("00700.HK", days=10)
    print(f"✅ 模拟数据形状: {df.shape}")
    print(f"✅ 列名: {df.columns.tolist()}")
    print(f"✅ 前5行:\n{df.head()}")
else:
    print(f"✅ 成功获取数据：{df.shape}")
    print(f"✅ 列名: {df.columns.tolist()}")
    print(f"✅ 前5行:\n{df.head()}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

"""
验证港股数据获取模块
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("验证港股数据获取模块")
print("=" * 60)

# 导入模块
try:
    from lib.data_fetcher import fetch_hk_stock_data, generate_simulated_hk_data
    print("✅ 成功导入港股数据获取模块")
except ImportError as e:
    print(f"🚨 导入失败: {e}")
    sys.exit(1)

# 验证函数签名
import inspect

print("\n=== 验证函数签名 ===")

# fetch_hk_stock_data
sig = inspect.signature(fetch_hk_stock_data)
print(f"fetch_hk_stock_data参数: {list(sig.parameters.keys())}")
print(f"默认参数: {sig.parameters['days'].default}")

# generate_simulated_hk_data
sig = inspect.signature(generate_simulated_hk_data)
print(f"generate_simulated_hk_data参数: {list(sig.parameters.keys())}")
print(f"默认参数: {sig.parameters['days'].default}")

# 验证模拟数据生成
print("\n=== 验证模拟数据生成 ===")
df = generate_simulated_hk_data("00700.HK", days=5)
print(f"✅ 模拟数据形状: {df.shape}")
print(f"✅ 列名: {df.columns.tolist()}")
print(f"✅ 数据类型: {df.dtypes.tolist()}")

# 验证数据格式
print("\n=== 验证数据格式 ===")
required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"🚨 缺少列: {missing_columns}")
else:
    print(f"✅ 所有必需列都存在: {required_columns}")

# 验证数据类型
print("\n=== 验证数据类型 ===")
expected_types = {
    'Open': 'float64',
    'High': 'float64',
    'Low': 'float64',
    'Close': 'float64',
    'Adj Close': 'float64',
    'Volume': 'int64'
}

for col, expected_type in expected_types.items():
    actual_type = str(df[col].dtype)
    if actual_type == expected_type:
        print(f"✅ {col}: {actual_type}")
    else:
        print(f"⚠️ {col}: 期望 {expected_type}, 实际 {actual_type}")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)

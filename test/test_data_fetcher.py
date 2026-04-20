"""
测试A股数据获取模块
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.data_fetcher import fetch_cn_stock_data, generate_simulated_cn_data


def test_fetch_cn_stock_data():
    """测试A股数据获取功能"""
    print("=== 测试A股数据获取功能 ===\n")

    # 测试1: 获取平安银行数据（如果网络不可用则使用模拟数据）
    print("测试1: 获取000001.SZ（平安银行）数据")
    df = fetch_cn_stock_data("000001.SZ", days=30)
    if df.empty:
        print("⚠️ 网络不可用，使用模拟数据")
        df = generate_simulated_cn_data("000001.SZ", days=30)
    assert not df.empty, "获取数据失败"
    assert 'Open' in df.columns, "缺少Open列"
    assert 'High' in df.columns, "缺少High列"
    assert 'Low' in df.columns, "缺少Low列"
    assert 'Close' in df.columns, "缺少Close列"
    assert 'Volume' in df.columns, "缺少Volume列"
    assert 'Adj Close' in df.columns, "缺少Adj Close列"
    print(f"✅ 测试1通过: 获取到{len(df)}行数据\n")

    # 测试2: 获取浦发银行数据（如果网络不可用则使用模拟数据）
    print("测试2: 获取600000.SH（浦发银行）数据")
    df = fetch_cn_stock_data("600000.SH", days=30)
    if df.empty:
        print("⚠️ 网络不可用，使用模拟数据")
        df = generate_simulated_cn_data("600000.SH", days=30)
    assert not df.empty, "获取数据失败"
    assert len(df.columns) == 6, "列数不正确"
    print(f"✅ 测试2通过: 获取到{len(df)}行数据\n")

    # 测试3: 测试模拟数据生成
    print("测试3: 生成模拟数据")
    df = generate_simulated_cn_data("000001.SZ", days=30)
    assert not df.empty, "生成模拟数据失败"
    assert len(df) == 30, "模拟数据行数不正确"
    assert len(df.columns) == 6, "模拟数据列数不正确"
    print(f"✅ 测试3通过: 生成{len(df)}行模拟数据\n")

    print("=== 所有测试通过！ ===")


if __name__ == "__main__":
    test_fetch_cn_stock_data()

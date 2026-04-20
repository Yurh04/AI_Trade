"""
统一数据获取接口
整合美股/A股/港股数据源，提供统一的股票数据获取接口
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
import traceback
from yfinance.exceptions import YFRateLimitError

# 导入数据获取模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.data_fetcher import (
    fetch_cn_stock_data,
    fetch_hk_stock_data,
    generate_simulated_cn_data,
    generate_simulated_hk_data
)

# 导入缓存模块
from lib.data_cache import get_cached_data, save_to_cache


def fetch_stock_data_unified(symbol: str, market: str, days: int = 30) -> pd.DataFrame:
    """
    统一股票数据获取接口

    Args:
        symbol: 股票代码
            - 美股: 如 "AAPL", "TSLA"
            - A股: 如 "000001.SZ", "600000.SH"
            - 港股: 如 "00700.HK", "00941.HK"
        market: 市场类型
            - "US": 美股（使用yfinance）
            - "CN": A股（使用akshare）
            - "HK": 港股（使用akshare）
        days: 获取天数，默认30天

    Returns:
        DataFrame: 包含Open/High/Low/Close/Adj Close/Volume列的股票数据
                  如果获取失败，返回模拟数据
    """
    # 验证market参数
    market = market.upper()
    if market not in ["US", "CN", "HK"]:
        print(f"🚨 不支持的市场类型: {market}，支持的市场: US, CN, HK")
        return pd.DataFrame()

    print(f"📊 获取股票数据: {symbol} ({market}市场, {days}天)")

    # 1. 尝试从缓存获取数据
    cached_data = get_cached_data(symbol, market, days)
    if cached_data is not None and not cached_data.empty:
        return cached_data

    # 2. 根据市场类型选择数据源
    data = None
    try:
        if market == "US":
            data = _fetch_us_stock_data(symbol, days)
        elif market == "CN":
            data = _fetch_cn_stock_data(symbol, days)
        elif market == "HK":
            data = _fetch_hk_stock_data(symbol, days)
    except Exception as e:
        print(f"🚨 获取数据时出错: {str(e)}")
        print(f"错误详情: {traceback.format_exc()}")

    # 3. 如果获取成功，保存到缓存
    if data is not None and not data.empty:
        # 确保数据格式统一
        data = _normalize_dataframe(data)
        save_to_cache(symbol, market, days, data)
        return data

    # 4. 如果获取失败，使用模拟数据
    print(f"⚠️ 无法获取实时数据，使用模拟数据...")
    simulated_data = _generate_simulated_data(symbol, market, days)
    return simulated_data


def _fetch_us_stock_data(symbol: str, days: int) -> pd.DataFrame:
    """
    获取美股数据（使用yfinance）

    Args:
        symbol: 股票代码
        days: 天数

    Returns:
        DataFrame: 美股数据
    """
    max_retries = 5
    initial_delay = 2
    backoff_factor = 2

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = initial_delay * (backoff_factor ** attempt) + random.uniform(0, attempt)
                print(f"⚠️ 速率限制，等待 {delay:.1f} 秒后重试 {attempt + 1}/{max_retries}...")
                time.sleep(delay)

            print(f"尝试 {attempt + 1}/{max_retries}: 获取 {symbol} 数据（美股）")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{days}d")

            if not data.empty:
                print(f"✅ 成功获取美股数据：{len(data)} 行，最新收盘价 {data['Close'].iloc[-1]}")
                return data

            info = ticker.info
            if info.get("delisted", False):
                print(f"⚠️ {symbol} 已退市")
                return pd.DataFrame()
            print("未返回数据，可能限流或其他原因，公司长名：", info.get("longName"))

        except YFRateLimitError:
            print(f"⚠️ 速率限制错误 ({attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"🚨 获取美股数据时出错 ({attempt + 1}/{max_retries}): {str(e)}")

    return pd.DataFrame()


def _fetch_cn_stock_data(symbol: str, days: int) -> pd.DataFrame:
    """
    获取A股数据（使用akshare）

    Args:
        symbol: 股票代码
        days: 天数

    Returns:
        DataFrame: A股数据
    """
    print(f"获取A股数据: {symbol}")
    try:
        data = fetch_cn_stock_data(symbol, days)
        if data.empty:
            print(f"⚠️ A股数据为空: {symbol}")
        return data
    except Exception as e:
        print(f"🚨 获取A股数据时出错: {str(e)}")
        return pd.DataFrame()


def _fetch_hk_stock_data(symbol: str, days: int) -> pd.DataFrame:
    """
    获取港股数据（使用akshare）

    Args:
        symbol: 股票代码
        days: 天数

    Returns:
        DataFrame: 港股数据
    """
    print(f"获取港股数据: {symbol}")
    try:
        data = fetch_hk_stock_data(symbol, days)
        if data.empty:
            print(f"⚠️ 港股数据为空: {symbol}")
        return data
    except Exception as e:
        print(f"🚨 获取港股数据时出错: {str(e)}")
        return pd.DataFrame()


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化DataFrame格式，确保包含所有必需的列

    Args:
        df: 原始DataFrame

    Returns:
        DataFrame: 标准化后的DataFrame
    """
    if df.empty:
        return df

    # 确保包含Adj Close列
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            print("⚠️ 数据中缺少'Adj Close'和'Close'列")

    # 确保列顺序一致
    required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    available_columns = [col for col in required_columns if col in df.columns]

    if available_columns:
        df = df[available_columns]

    return df


def _generate_simulated_data(symbol: str, market: str, days: int) -> pd.DataFrame:
    """
    生成模拟数据（fallback机制）

    Args:
        symbol: 股票代码
        market: 市场类型
        days: 天数

    Returns:
        DataFrame: 模拟的股票数据
    """
    print(f"📊 生成模拟数据: {symbol} ({market}市场, {days}天)")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # 生成交易日（排除周末）
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # 根据市场设置基础价格
    base_prices = {
        'US': {
            'AAPL': 170, 'TSLA': 180, 'GOOGL': 120, 'MSFT': 350,
            'AMZN': 150, 'META': 300, 'NVDA': 500, 'NFLX': 400
        },
        'CN': {
            '000001': 10.0, '600000': 7.0, '600519': 1800.0,
            '000858': 25.0, '000002': 20.0, '600036': 35.0
        },
        'HK': {
            '00700': 350.0, '00941': 150.0, '01299': 80.0,
            '03690': 100.0, '09618': 150.0, '02318': 200.0
        }
    }

    # 清理股票代码（移除后缀）
    clean_symbol = symbol.replace('.SZ', '').replace('.SH', '').replace('.HK', '')

    # 获取基础价格
    market_prices = base_prices.get(market, {})
    base = market_prices.get(clean_symbol, 100.0)

    # 根据市场调整成交量范围
    volume_ranges = {
        'US': (10000000, 80000000),  # 美股
        'CN': (1000000, 50000000),   # A股
        'HK': (500000, 20000000)     # 港股
    }
    volume_range = volume_ranges.get(market, (1000000, 50000000))

    data = []
    price = base

    for date in dates:
        # 随机波动
        change = random.uniform(-0.04, 0.04)
        price *= (1 + change)

        # 生成OHLC数据
        o = price * random.uniform(0.99, 1.01)
        h = price * random.uniform(1.002, 1.025)
        l = price * random.uniform(0.975, 0.998)
        v = random.randint(volume_range[0], volume_range[1])

        data.append({
            'Open': round(o, 2),
            'High': round(h, 2),
            'Low': round(l, 2),
            'Close': round(price, 2),
            'Adj Close': round(price, 2),
            'Volume': v
        })

    df = pd.DataFrame(data, index=dates)
    print(f"✅ 模拟数据生成完毕，共 {len(df)} 行")
    return df


if __name__ == "__main__":
    # 测试代码
    print("测试统一数据获取接口...")

    # 测试美股
    print("\n=== 测试美股 (AAPL) ===")
    df = fetch_stock_data_unified("AAPL", "US", days=30)
    if not df.empty:
        print(f"✅ 数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前5行:\n{df.head()}")
    else:
        print("❌ 获取失败")

    # 测试A股
    print("\n=== 测试A股 (000001.SZ) ===")
    df = fetch_stock_data_unified("000001.SZ", "CN", days=30)
    if not df.empty:
        print(f"✅ 数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前5行:\n{df.head()}")
    else:
        print("❌ 获取失败")

    # 测试港股
    print("\n=== 测试港股 (00700.HK) ===")
    df = fetch_stock_data_unified("00700.HK", "HK", days=30)
    if not df.empty:
        print(f"✅ 数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前5行:\n{df.head()}")
    else:
        print("❌ 获取失败")

    # 测试缓存
    print("\n=== 测试缓存功能 ===")
    print("再次获取AAPL数据（应该从缓存读取）...")
    df = fetch_stock_data_unified("AAPL", "US", days=30)
    if not df.empty:
        print(f"✅ 从缓存获取数据: {df.shape}")

    print("\n✅ 测试完成")

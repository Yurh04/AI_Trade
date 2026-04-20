"""
A股数据获取模块
使用Akshare库获取A股日线数据
"""

import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import time
import random
import traceback


def fetch_cn_stock_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """
    获取A股日线数据

    Args:
        symbol: A股股票代码（如000001.SZ, 600000.SH）
        days: 获取天数，默认30天

    Returns:
        DataFrame: 包含Open/High/Low/Close/Volume/Adj Close列的股票数据
    """
    max_retries = 5
    initial_delay = 2
    backoff_factor = 2

    # 处理股票代码格式
    # Akshare使用不带后缀的代码（如000001, 600000）
    clean_symbol = symbol.replace('.SZ', '').replace('.SH', '')

    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 10)  # 多取几天以确保有足够交易日

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = initial_delay * (backoff_factor ** attempt) + random.uniform(0, attempt)
                print(f"⚠️ 速率限制，等待 {delay:.1f} 秒后重试 {attempt + 1}/{max_retries}...")
                time.sleep(delay)

            print(f"尝试 {attempt + 1}/{max_retries}: 获取 {symbol} 数据")

            # 使用Akshare获取A股数据
            data = ak.stock_zh_a_hist(
                symbol=clean_symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq"  # 前复权
            )

            if data is not None and not data.empty:
                # Akshare返回的列名：日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率
                # 需要映射到yfinance格式：Open, High, Low, Close, Volume, Adj Close

                # 重命名列
                column_mapping = {
                    '日期': 'Date',
                    '开盘': 'Open',
                    '收盘': 'Close',
                    '最高': 'High',
                    '最低': 'Low',
                    '成交量': 'Volume'
                }

                data = data.rename(columns=column_mapping)

                # 设置日期为索引
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    data.set_index('Date', inplace=True)

                # 添加Adj Close列（A股使用前复权，Close即为Adj Close）
                if 'Close' in data.columns:
                    data['Adj Close'] = data['Close']

                # 确保列顺序与yfinance一致
                required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                available_columns = [col for col in required_columns if col in data.columns]
                data = data[available_columns]

                # 只返回指定天数的数据
                data = data.tail(days)

                print(f"✅ 成功获取数据：{len(data)} 行，最新收盘价 {data['Close'].iloc[-1]}")
                return data
            else:
                print(f"⚠️ 未获取到 {symbol} 的数据")

        except Exception as e:
            print(f"🚨 获取数据时出错 ({attempt + 1}/{max_retries}): {str(e)}")
            print(f"错误详情: {traceback.format_exc()}")

    print(f"⚠️ 无法获取 {symbol} 的实时数据，返回空DataFrame")
    return pd.DataFrame()


def generate_simulated_cn_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """
    生成模拟A股数据（用于测试或API不可用时）

    Args:
        symbol: A股股票代码
        days: 生成天数

    Returns:
        DataFrame: 模拟的股票数据
    """
    print(f"📊 生成模拟A股数据: {symbol} ({days} 天)")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # 生成交易日（排除周末）
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # 基础价格（根据股票代码设置不同基准）
    base_prices = {
        '000001': 10.0,  # 平安银行
        '600000': 7.0,   # 浦发银行
        '600519': 1800.0,  # 贵州茅台
        '000858': 25.0,  # 五粮液
    }

    clean_symbol = symbol.replace('.SZ', '').replace('.SH', '')
    base = base_prices.get(clean_symbol, 10.0)

    data = []
    price = base

    for date in dates:
        # 随机波动
        change = random.uniform(-0.03, 0.03)
        price *= (1 + change)

        # 生成OHLC数据
        o = price * random.uniform(0.99, 1.01)
        h = price * random.uniform(1.002, 1.02)
        l = price * random.uniform(0.98, 0.998)
        v = random.randint(1000000, 50000000)  # A股成交量通常较大

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


def fetch_hk_stock_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """
    获取港股日线数据

    Args:
        symbol: 港股股票代码（如00700.HK, 00941.HK）
        days: 获取天数，默认30天

    Returns:
        DataFrame: 包含Open/High/Low/Close/Volume/Adj Close列的股票数据
    """
    max_retries = 5
    initial_delay = 2
    backoff_factor = 2

    # 处理股票代码格式
    # Akshare使用不带后缀的代码（如00700, 00941）
    clean_symbol = symbol.replace('.HK', '')

    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 10)  # 多取几天以确保有足够交易日

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = initial_delay * (backoff_factor ** attempt) + random.uniform(0, attempt)
                print(f"⚠️ 速率限制，等待 {delay:.1f} 秒后重试 {attempt + 1}/{max_retries}...")
                time.sleep(delay)

            print(f"尝试 {attempt + 1}/{max_retries}: 获取 {symbol} 数据")

            # 使用Akshare获取港股数据
            data = ak.stock_hk_hist(
                symbol=clean_symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq"  # 前复权
            )

            if data is not None and not data.empty:
                # Akshare返回的列名：日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率
                # 需要映射到yfinance格式：Open, High, Low, Close, Volume, Adj Close

                # 重命名列
                column_mapping = {
                    '日期': 'Date',
                    '开盘': 'Open',
                    '收盘': 'Close',
                    '最高': 'High',
                    '最低': 'Low',
                    '成交量': 'Volume'
                }

                data = data.rename(columns=column_mapping)

                # 设置日期为索引
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    data.set_index('Date', inplace=True)

                # 添加Adj Close列（港股使用前复权，Close即为Adj Close）
                if 'Close' in data.columns:
                    data['Adj Close'] = data['Close']

                # 确保列顺序与yfinance一致
                required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                available_columns = [col for col in required_columns if col in data.columns]
                data = data[available_columns]

                # 只返回指定天数的数据
                data = data.tail(days)

                print(f"✅ 成功获取数据：{len(data)} 行，最新收盘价 {data['Close'].iloc[-1]}")
                return data
            else:
                print(f"⚠️ 未获取到 {symbol} 的数据")

        except Exception as e:
            print(f"🚨 获取数据时出错 ({attempt + 1}/{max_retries}): {str(e)}")
            print(f"错误详情: {traceback.format_exc()}")

    print(f"⚠️ 无法获取 {symbol} 的实时数据，返回空DataFrame")
    return pd.DataFrame()


def generate_simulated_hk_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """
    生成模拟港股数据（用于测试或API不可用时）

    Args:
        symbol: 港股股票代码
        days: 生成天数

    Returns:
        DataFrame: 模拟的股票数据
    """
    print(f"📊 生成模拟港股数据: {symbol} ({days} 天)")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # 生成交易日（排除周末）
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # 基础价格（根据股票代码设置不同基准）
    base_prices = {
        '00700': 350.0,  # 腾讯控股
        '00941': 150.0,  # 中国移动
        '01299': 80.0,   # 友邦保险
        '03690': 100.0,  # 美团
        '09618': 150.0,  # 京东集团
    }

    clean_symbol = symbol.replace('.HK', '')
    base = base_prices.get(clean_symbol, 100.0)

    data = []
    price = base

    for date in dates:
        # 随机波动
        change = random.uniform(-0.03, 0.03)
        price *= (1 + change)

        # 生成OHLC数据
        o = price * random.uniform(0.99, 1.01)
        h = price * random.uniform(1.002, 1.02)
        l = price * random.uniform(0.98, 0.998)
        v = random.randint(500000, 20000000)  # 港股成交量

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
    print("测试A股数据获取功能...")

    # 测试平安银行（000001.SZ）
    print("\n=== 测试 000001.SZ（平安银行）===")
    df = fetch_cn_stock_data("000001.SZ", days=30)
    if not df.empty:
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前5行:\n{df.head()}")
    else:
        print("获取失败，使用模拟数据")
        df = generate_simulated_cn_data("000001.SZ", days=30)
        print(f"模拟数据形状: {df.shape}")
        print(f"前5行:\n{df.head()}")

    # 测试浦发银行（600000.SH）
    print("\n=== 测试 600000.SH（浦发银行）===")
    df = fetch_cn_stock_data("600000.SH", days=30)
    if not df.empty:
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前5行:\n{df.head()}")
    else:
        print("获取失败，使用模拟数据")
        df = generate_simulated_cn_data("600000.SH", days=30)
        print(f"模拟数据形状: {df.shape}")
        print(f"前5行:\n{df.head()}")

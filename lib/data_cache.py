"""
数据缓存模块 - 股票数据缓存管理
提供股票数据的缓存查询、更新和过期检查功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from lib.db import save_stock_cache, get_stock_cache


def dataframe_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """
    将DataFrame转换为可JSON序列化的字典格式

    Args:
        df: pandas DataFrame对象

    Returns:
        Dict[str, Any]: 包含DataFrame数据的字典
    """
    if df.empty:
        return {}

    # 将DataFrame转换为字典，保留索引
    data_dict = {
        'index': df.index.strftime('%Y-%m-%d').tolist(),
        'columns': df.columns.tolist(),
        'data': df.values.tolist()
    }

    return data_dict


def dict_to_dataframe(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    将字典格式转换回DataFrame

    Args:
        data_dict: 包含DataFrame数据的字典

    Returns:
        pd.DataFrame: 重建的DataFrame对象
    """
    if not data_dict or 'data' not in data_dict:
        return pd.DataFrame()

    # 从字典重建DataFrame
    df = pd.DataFrame(
        data_dict['data'],
        index=pd.to_datetime(data_dict['index']),
        columns=data_dict['columns']
    )

    return df


def get_cached_data(symbol: str, market: str, days: int, cache_ttl: int = 3600) -> Optional[pd.DataFrame]:
    """
    从缓存获取股票数据，检查是否过期

    Args:
        symbol: 股票代码
        market: 市场类型（如 'US', 'CN'）
        days: 数据天数
        cache_ttl: 缓存过期时间（秒），默认3600秒（1小时）

    Returns:
        Optional[pd.DataFrame]: 缓存的DataFrame数据，如果不存在或已过期返回None
    """
    try:
        # 从数据库获取缓存
        cache_result = get_stock_cache(symbol, market, days)

        if not cache_result:
            print(f"📭 缓存未命中: {symbol}_{market}_{days}")
            return None

        # 检查缓存是否过期
        cache_timestamp = datetime.fromisoformat(cache_result['timestamp'])
        current_time = datetime.now()
        cache_age = (current_time - cache_timestamp).total_seconds()

        if cache_age > cache_ttl:
            print(f"⏰ 缓存已过期: {symbol}_{market}_{days} (缓存时长: {cache_age:.0f}秒, TTL: {cache_ttl}秒)")
            return None

        print(f"✅ 缓存命中: {symbol}_{market}_{days} (缓存时长: {cache_age:.0f}秒)")

        # 将字典转换回DataFrame
        df = dict_to_dataframe(cache_result['data'])
        return df

    except Exception as e:
        print(f"🚨 获取缓存数据失败: {str(e)}")
        return None


def save_to_cache(symbol: str, market: str, days: int, data: pd.DataFrame) -> bool:
    """
    将股票数据保存到缓存

    Args:
        symbol: 股票代码
        market: 市场类型（如 'US', 'CN'）
        days: 数据天数
        data: pandas DataFrame格式的股票数据

    Returns:
        bool: 是否保存成功
    """
    try:
        if data.empty:
            print(f"⚠️ 数据为空，跳过缓存: {symbol}_{market}_{days}")
            return False

        # 将DataFrame转换为可JSON序列化的字典
        data_dict = dataframe_to_dict(data)

        # 保存到数据库
        success = save_stock_cache(symbol, market, days, data_dict)

        if success:
            print(f"💾 数据已缓存: {symbol}_{market}_{days} ({len(data)} 行)")
        else:
            print(f"❌ 缓存保存失败: {symbol}_{market}_{days}")

        return success

    except Exception as e:
        print(f"🚨 保存缓存数据失败: {str(e)}")
        return False


def is_cache_valid(symbol: str, market: str, days: int, cache_ttl: int = 3600) -> bool:
    """
    检查缓存是否有效（存在且未过期）

    Args:
        symbol: 股票代码
        market: 市场类型（如 'US', 'CN'）
        days: 数据天数
        cache_ttl: 缓存过期时间（秒），默认3600秒（1小时）

    Returns:
        bool: 缓存是否有效
    """
    try:
        cache_result = get_stock_cache(symbol, market, days)

        if not cache_result:
            return False

        # 检查缓存是否过期
        cache_timestamp = datetime.fromisoformat(cache_result['timestamp'])
        current_time = datetime.now()
        cache_age = (current_time - cache_timestamp).total_seconds()

        return cache_age <= cache_ttl

    except Exception as e:
        print(f"🚨 检查缓存有效性失败: {str(e)}")
        return False


def get_cache_age(symbol: str, market: str, days: int) -> Optional[float]:
    """
    获取缓存的年龄（秒）

    Args:
        symbol: 股票代码
        market: 市场类型（如 'US', 'CN'）
        days: 数据天数

    Returns:
        Optional[float]: 缓存年龄（秒），如果缓存不存在返回None
    """
    try:
        cache_result = get_stock_cache(symbol, market, days)

        if not cache_result:
            return None

        cache_timestamp = datetime.fromisoformat(cache_result['timestamp'])
        current_time = datetime.now()
        cache_age = (current_time - cache_timestamp).total_seconds()

        return cache_age

    except Exception as e:
        print(f"🚨 获取缓存年龄失败: {str(e)}")
        return None


if __name__ == '__main__':
    # 测试代码
    print("测试数据缓存模块...")

    # 创建测试数据
    test_data = pd.DataFrame({
        'Open': [100.0, 101.0, 102.0],
        'High': [105.0, 106.0, 107.0],
        'Low': [99.0, 100.0, 101.0],
        'Close': [104.0, 105.0, 106.0],
        'Volume': [1000000, 1100000, 1200000]
    }, index=pd.date_range('2024-01-01', periods=3))

    print(f"\n测试数据:\n{test_data}")

    # 测试保存到缓存
    print("\n测试保存到缓存...")
    save_result = save_to_cache('AAPL', 'US', 30, test_data)
    print(f"保存结果: {save_result}")

    # 测试从缓存获取
    print("\n测试从缓存获取...")
    cached_data = get_cached_data('AAPL', 'US', 30, cache_ttl=3600)
    if cached_data is not None:
        print(f"缓存数据:\n{cached_data}")
    else:
        print("未获取到缓存数据")

    # 测试缓存有效性检查
    print("\n测试缓存有效性检查...")
    is_valid = is_cache_valid('AAPL', 'US', 30, cache_ttl=3600)
    print(f"缓存是否有效: {is_valid}")

    # 测试获取缓存年龄
    print("\n测试获取缓存年龄...")
    age = get_cache_age('AAPL', 'US', 30)
    if age is not None:
        print(f"缓存年龄: {age:.2f} 秒")
    else:
        print("未获取到缓存年龄")

    print("\n✅ 测试完成")

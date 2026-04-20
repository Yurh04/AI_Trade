"""
数据库模块 - SQLite数据库初始化和操作
提供股票数据缓存、历史分析和用户偏好的持久化存储
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'analysis_memory.db')


def get_db_connection() -> sqlite3.Connection:
    """
    获取数据库连接

    Returns:
        sqlite3.Connection: 数据库连接对象
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    初始化数据库，创建所有表
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            days INTEGER NOT NULL,
            data TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            cache_key TEXT UNIQUE NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            analysis TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            preferences TEXT NOT NULL,
            updated_at DATETIME NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_stock_cache_key ON stock_cache(cache_key)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_analysis_history_symbol ON analysis_history(symbol)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_analysis_history_timestamp ON analysis_history(timestamp)
    ''')

    conn.commit()
    conn.close()
    print("✅ 数据库初始化完成")


# ==================== stock_cache 表操作 ====================

def save_stock_cache(symbol: str, market: str, days: int, data: Dict[str, Any]) -> bool:
    """
    保存股票数据到缓存

    Args:
        symbol: 股票代码
        market: 市场类型
        days: 数据天数
        data: 股票数据字典

    Returns:
        bool: 是否保存成功
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cache_key = f"{symbol}_{market}_{days}"
        timestamp = datetime.now().isoformat()
        data_json = json.dumps(data, ensure_ascii=False)

        cursor.execute('''
            INSERT OR REPLACE INTO stock_cache
            (symbol, market, days, data, timestamp, cache_key)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, market, days, data_json, timestamp, cache_key))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"🚨 保存股票缓存失败: {str(e)}")
        return False


def get_stock_cache(symbol: str, market: str, days: int) -> Optional[Dict[str, Any]]:
    """
    从缓存获取股票数据

    Args:
        symbol: 股票代码
        market: 市场类型
        days: 数据天数

    Returns:
        Optional[Dict[str, Any]]: 缓存的数据，如果不存在返回None
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cache_key = f"{symbol}_{market}_{days}"
        cursor.execute('''
            SELECT data, timestamp FROM stock_cache
            WHERE cache_key = ?
        ''', (cache_key,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'data': json.loads(row['data']),
                'timestamp': row['timestamp']
            }
        return None
    except Exception as e:
        print(f"🚨 获取股票缓存失败: {str(e)}")
        return None


def clear_stock_cache(symbol: Optional[str] = None) -> bool:
    """
    清除股票缓存

    Args:
        symbol: 股票代码，如果为None则清除所有缓存

    Returns:
        bool: 是否清除成功
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if symbol:
            cursor.execute('DELETE FROM stock_cache WHERE symbol = ?', (symbol,))
        else:
            cursor.execute('DELETE FROM stock_cache')

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"🚨 清除股票缓存失败: {str(e)}")
        return False


# ==================== analysis_history 表操作 ====================

def save_analysis_history(symbol: str, market: str, analysis: Dict[str, Any]) -> bool:
    """
    保存分析历史记录

    Args:
        symbol: 股票代码
        market: 市场类型
        analysis: 分析结果字典

    Returns:
        bool: 是否保存成功
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        analysis_json = json.dumps(analysis, ensure_ascii=False)

        cursor.execute('''
            INSERT INTO analysis_history (symbol, market, analysis, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (symbol, market, analysis_json, timestamp))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"🚨 保存分析历史失败: {str(e)}")
        return False


def get_analysis_history(symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    获取股票的分析历史记录

    Args:
        symbol: 股票代码
        limit: 返回记录数量限制

    Returns:
        List[Dict[str, Any]]: 分析历史记录列表
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT symbol, market, analysis, timestamp
            FROM analysis_history
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (symbol, limit))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'symbol': row['symbol'],
                'market': row['market'],
                'analysis': json.loads(row['analysis']),
                'timestamp': row['timestamp']
            }
            for row in rows
        ]
    except Exception as e:
        print(f"🚨 获取分析历史失败: {str(e)}")
        return []


def clear_analysis_history(symbol: Optional[str] = None) -> bool:
    """
    清除分析历史记录

    Args:
        symbol: 股票代码，如果为None则清除所有记录

    Returns:
        bool: 是否清除成功
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if symbol:
            cursor.execute('DELETE FROM analysis_history WHERE symbol = ?', (symbol,))
        else:
            cursor.execute('DELETE FROM analysis_history')

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"🚨 清除分析历史失败: {str(e)}")
        return False


# ==================== user_preferences 表操作 ====================

def save_user_preferences(user_id: str, preferences: Dict[str, Any]) -> bool:
    """
    保存用户偏好设置

    Args:
        user_id: 用户ID
        preferences: 偏好设置字典

    Returns:
        bool: 是否保存成功
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        updated_at = datetime.now().isoformat()
        preferences_json = json.dumps(preferences, ensure_ascii=False)

        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences (user_id, preferences, updated_at)
            VALUES (?, ?, ?)
        ''', (user_id, preferences_json, updated_at))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"🚨 保存用户偏好失败: {str(e)}")
        return False


def get_user_preferences(user_id: str) -> Optional[Dict[str, Any]]:
    """
    获取用户偏好设置

    Args:
        user_id: 用户ID

    Returns:
        Optional[Dict[str, Any]]: 用户偏好设置，如果不存在返回None
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT preferences, updated_at FROM user_preferences
            WHERE user_id = ?
        ''', (user_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'preferences': json.loads(row['preferences']),
                'updated_at': row['updated_at']
            }
        return None
    except Exception as e:
        print(f"🚨 获取用户偏好失败: {str(e)}")
        return None


def delete_user_preferences(user_id: str) -> bool:
    """
    删除用户偏好设置

    Args:
        user_id: 用户ID

    Returns:
        bool: 是否删除成功
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM user_preferences WHERE user_id = ?', (user_id,))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"🚨 删除用户偏好失败: {str(e)}")
        return False


# ==================== 数据库维护操作 ====================

def get_db_stats() -> Dict[str, Any]:
    """
    获取数据库统计信息

    Returns:
        Dict[str, Any]: 包含各表记录数的统计信息
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        stats = {}

        cursor.execute('SELECT COUNT(*) as count FROM stock_cache')
        stats['stock_cache_count'] = cursor.fetchone()['count']

        cursor.execute('SELECT COUNT(*) as count FROM analysis_history')
        stats['analysis_history_count'] = cursor.fetchone()['count']

        cursor.execute('SELECT COUNT(*) as count FROM user_preferences')
        stats['user_preferences_count'] = cursor.fetchone()['count']

        if os.path.exists(DB_PATH):
            stats['db_size_bytes'] = os.path.getsize(DB_PATH)
            stats['db_size_mb'] = round(stats['db_size_bytes'] / (1024 * 1024), 2)

        conn.close()
        return stats
    except Exception as e:
        print(f"🚨 获取数据库统计失败: {str(e)}")
        return {}


def cleanup_old_cache(days: int = 7) -> int:
    """
    清理指定天数之前的缓存数据

    Args:
        days: 保留天数

    Returns:
        int: 清理的记录数
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute('''
            DELETE FROM stock_cache
            WHERE timestamp < ?
        ''', (cutoff_date,))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        print(f"✅ 清理了 {deleted_count} 条过期缓存记录")
        return deleted_count
    except Exception as e:
        print(f"🚨 清理过期缓存失败: {str(e)}")
        return 0


if __name__ == '__main__':
    print("测试数据库初始化...")
    init_db()

    print("\n测试基本操作...")
    save_stock_cache('AAPL', 'US', 30, {'price': 150.0, 'volume': 1000000})
    cache_data = get_stock_cache('AAPL', 'US', 30)
    print(f"缓存数据: {cache_data}")

    save_analysis_history('AAPL', 'US', {'recommendation': '买入', 'score': 8.5})
    history = get_analysis_history('AAPL', limit=1)
    print(f"分析历史: {history}")

    save_user_preferences('user_001', {'theme': 'dark', 'language': 'zh'})
    prefs = get_user_preferences('user_001')
    print(f"用户偏好: {prefs}")

    stats = get_db_stats()
    print(f"\n数据库统计: {stats}")

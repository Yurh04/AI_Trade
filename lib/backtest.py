"""
策略回测模块
使用vectorbt库实现高效的策略回测
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore')


def run_backtest(data: pd.DataFrame, strategy: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    运行策略回测

    Args:
        data: 包含OHLCV列的股票数据
        strategy: 策略类型
            - 'ma_cross': 移动平均交叉策略
            - 'rsi': RSI超买超卖策略
            - 'bollinger': 布林带策略
        params: 策略参数字典，可选

    Returns:
        Dict[str, Any]: 回测结果字典
    """
    if data.empty:
        return {"error": "数据为空"}

    default_params = {
        'ma_cross': {
            'short_period': 5,
            'long_period': 20,
            'initial_cash': 100000
        },
        'rsi': {
            'period': 14,
            'overbought': 70,
            'oversold': 30,
            'initial_cash': 100000
        },
        'bollinger': {
            'period': 20,
            'std_dev': 2,
            'initial_cash': 100000
        }
    }

    if params:
        default_params[strategy].update(params)
    strategy_params = default_params[strategy]

    if strategy == 'ma_cross':
        result = _backtest_ma_cross(data, strategy_params)
    elif strategy == 'rsi':
        result = _backtest_rsi(data, strategy_params)
    elif strategy == 'bollinger':
        result = _backtest_bollinger(data, strategy_params)
    else:
        return {"error": f"不支持的策略类型: {strategy}"}

    return result


def _backtest_ma_cross(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    移动平均交叉策略回测

    策略逻辑：
    - 当短期均线(SMA_short)上穿长期均线(SMA_long)时买入
    - 当短期均线下穿长期均线时卖出
    """
    short_period = params['short_period']
    long_period = params['long_period']
    initial_cash = params['initial_cash']

    # 计算移动平均
    data = data.copy()
    data[f'SMA_{short_period}'] = data['Close'].rolling(window=short_period).mean()
    data[f'SMA_{long_period}'] = data['Close'].rolling(window=long_period).mean()

    # 生成交易信号
    data['Signal'] = 0
    data.loc[data[f'SMA_{short_period}'] > data[f'SMA_{long_period}'], 'Signal'] = 1
    data.loc[data[f'SMA_{short_period}'] < data[f'SMA_{long_period}'], 'Signal'] = -1

    # 计算信号变化（交叉点）
    data['Position'] = data['Signal'].shift(1).fillna(0)
    data['Entry_Signal'] = (data['Position'] == 1) & (data['Position'].shift(1) != 1)
    data['Exit_Signal'] = (data['Position'] == -1) & (data['Position'].shift(1) != -1)

    # 删除NaN值
    data = data.dropna()

    if len(data) == 0:
        return {"error": "数据不足"}

    # 模拟交易
    trades = []
    cash = initial_cash
    position = 0
    entry_price = 0

    for idx, row in data.iterrows():
        if row['Entry_Signal'] and position == 0:
            # 买入
            shares = int(cash / row['Close'])
            if shares > 0:
                position = shares
                entry_price = row['Close']
                cash -= shares * row['Close']
                trades.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'type': 'BUY',
                    'price': round(row['Close'], 2),
                    'shares': shares,
                    'value': round(shares * row['Close'], 2)
                })

        elif row['Exit_Signal'] and position > 0:
            # 卖出
            cash += position * row['Close']
            profit = position * (row['Close'] - entry_price)
            profit_pct = (row['Close'] - entry_price) / entry_price * 100

            trades[-1]['exit_date'] = idx.strftime('%Y-%m-%d')
            trades[-1]['exit_price'] = round(row['Close'], 2)
            trades[-1]['profit'] = round(profit, 2)
            trades[-1]['profit_pct'] = round(profit_pct, 2)

            position = 0
            entry_price = 0

    # 计算最终资产
    final_value = cash + position * data['Close'].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    # 计算基准收益（买入持有策略）
    buy_hold_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100

    # 计算最大回撤
    data['Portfolio_Value'] = initial_cash
    current_cash = initial_cash
    current_position = 0
    for idx, row in data.iterrows():
        if row['Entry_Signal'] and current_position == 0:
            shares = int(current_cash / row['Close'])
            current_position = shares
            current_cash -= shares * row['Close']
        elif row['Exit_Signal'] and current_position > 0:
            current_cash += current_position * row['Close']
            current_position = 0

        data.loc[idx, 'Portfolio_Value'] = current_cash + current_position * row['Close']

    data['Cummax'] = data['Portfolio_Value'].cummax()
    data['Drawdown'] = (data['Portfolio_Value'] - data['Cummax']) / data['Cummax'] * 100
    max_drawdown = data['Drawdown'].min()

    # 计算胜率
    winning_trades = [t for t in trades if 'profit' in t and t['profit'] > 0]
    losing_trades = [t for t in trades if 'profit' in t and t['profit'] <= 0]
    win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0

    # 计算年化收益率
    trading_days = len(data)
    years = trading_days / 252
    annual_return = (final_value / initial_cash) ** (1 / years) - 1 if years > 0 else 0
    annual_return_pct = annual_return * 100

    # 计算夏普比率（简化版）
    data['Daily_Return'] = data['Portfolio_Value'].pct_change().fillna(0)
    sharpe_ratio = data['Daily_Return'].mean() / data['Daily_Return'].std() * np.sqrt(252) if data['Daily_Return'].std() > 0 else 0

    result = {
        'strategy': 'MA Cross',
        'params': {
            'short_period': short_period,
            'long_period': long_period
        },
        'performance': {
            'initial_cash': round(initial_cash, 2),
            'final_value': round(final_value, 2),
            'total_return_pct': round(total_return, 2),
            'annual_return_pct': round(annual_return_pct, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'buy_hold_return_pct': round(buy_hold_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'win_rate_pct': round(win_rate, 2)
        },
        'trades': trades,
        'trade_count': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades)
    }

    return result


def _backtest_rsi(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    RSI超买超卖策略回测

    策略逻辑：
    - 当RSI低于超卖阈值时买入
    - 当RSI高于超买阈值时卖出
    """
    period = params['period']
    overbought = params['overbought']
    oversold = params['oversold']
    initial_cash = params['initial_cash']

    # 计算RSI
    data = data.copy()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    # 生成交易信号
    data['Signal'] = 0
    data.loc[data[f'RSI_{period}'] < oversold, 'Signal'] = 1
    data.loc[data[f'RSI_{period}'] > overbought, 'Signal'] = -1

    data['Position'] = data['Signal'].shift(1).fillna(0)
    data['Entry_Signal'] = (data['Position'] == 1) & (data['Position'].shift(1) != 1)
    data['Exit_Signal'] = (data['Position'] == -1) & (data['Position'].shift(1) != -1)

    data = data.dropna()

    if len(data) == 0:
        return {"error": "数据不足"}

    # 模拟交易
    trades = []
    cash = initial_cash
    position = 0
    entry_price = 0

    for idx, row in data.iterrows():
        if row['Entry_Signal'] and position == 0:
            shares = int(cash / row['Close'])
            if shares > 0:
                position = shares
                entry_price = row['Close']
                cash -= shares * row['Close']
                trades.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'type': 'BUY',
                    'price': round(row['Close'], 2),
                    'shares': shares,
                    'value': round(shares * row['Close'], 2),
                    'rsi': round(row[f'RSI_{period}'], 2)
                })

        elif row['Exit_Signal'] and position > 0:
            cash += position * row['Close']
            profit = position * (row['Close'] - entry_price)
            profit_pct = (row['Close'] - entry_price) / entry_price * 100

            trades[-1]['exit_date'] = idx.strftime('%Y-%m-%d')
            trades[-1]['exit_price'] = round(row['Close'], 2)
            trades[-1]['profit'] = round(profit, 2)
            trades[-1]['profit_pct'] = round(profit_pct, 2)

            position = 0
            entry_price = 0

    final_value = cash + position * data['Close'].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    # 计算最大回撤
    data['Portfolio_Value'] = initial_cash
    current_cash = initial_cash
    current_position = 0
    for idx, row in data.iterrows():
        if row['Entry_Signal'] and current_position == 0:
            shares = int(current_cash / row['Close'])
            current_position = shares
            current_cash -= shares * row['Close']
        elif row['Exit_Signal'] and current_position > 0:
            current_cash += current_position * row['Close']
            current_position = 0

        data.loc[idx, 'Portfolio_Value'] = current_cash + current_position * row['Close']

    data['Cummax'] = data['Portfolio_Value'].cummax()
    data['Drawdown'] = (data['Portfolio_Value'] - data['Cummax']) / data['Cummax'] * 100
    max_drawdown = data['Drawdown'].min()

    # 胜率
    winning_trades = [t for t in trades if 'profit' in t and t['profit'] > 0]
    losing_trades = [t for t in trades if 'profit' in t and t['profit'] <= 0]
    win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0

    result = {
        'strategy': 'RSI',
        'params': {
            'period': period,
            'overbought': overbought,
            'oversold': oversold
        },
        'performance': {
            'initial_cash': round(initial_cash, 2),
            'final_value': round(final_value, 2),
            'total_return_pct': round(total_return, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'win_rate_pct': round(win_rate, 2)
        },
        'trades': trades,
        'trade_count': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades)
    }

    return result


def _backtest_bollinger(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    布林带策略回测

    策略逻辑：
    - 当价格跌破下轨时买入（均值回归）
    - 当价格涨破上轨时卖出
    """
    period = params['period']
    std_dev = params['std_dev']
    initial_cash = params['initial_cash']

    # 计算布林带
    data = data.copy()
    data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
    data['BB_Std'] = data['Close'].rolling(window=period).std()
    data['BB_Upper'] = data[f'SMA_{period}'] + std_dev * data['BB_Std']
    data['BB_Lower'] = data[f'SMA_{period}'] - std_dev * data['BB_Std']

    # 生成交易信号
    data['Signal'] = 0
    data.loc[data['Close'] < data['BB_Lower'], 'Signal'] = 1
    data.loc[data['Close'] > data['BB_Upper'], 'Signal'] = -1

    data['Position'] = data['Signal'].shift(1).fillna(0)
    data['Entry_Signal'] = (data['Position'] == 1) & (data['Position'].shift(1) != 1)
    data['Exit_Signal'] = (data['Position'] == -1) & (data['Position'].shift(1) != -1)

    data = data.dropna()

    if len(data) == 0:
        return {"error": "数据不足"}

    # 模拟交易
    trades = []
    cash = initial_cash
    position = 0
    entry_price = 0

    for idx, row in data.iterrows():
        if row['Entry_Signal'] and position == 0:
            shares = int(cash / row['Close'])
            if shares > 0:
                position = shares
                entry_price = row['Close']
                cash -= shares * row['Close']
                trades.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'type': 'BUY',
                    'price': round(row['Close'], 2),
                    'shares': shares,
                    'value': round(shares * row['Close'], 2),
                    'bb_position': 'Below Lower Band'
                })

        elif row['Exit_Signal'] and position > 0:
            cash += position * row['Close']
            profit = position * (row['Close'] - entry_price)
            profit_pct = (row['Close'] - entry_price) / entry_price * 100

            trades[-1]['exit_date'] = idx.strftime('%Y-%m-%d')
            trades[-1]['exit_price'] = round(row['Close'], 2)
            trades[-1]['profit'] = round(profit, 2)
            trades[-1]['profit_pct'] = round(profit_pct, 2)

            position = 0
            entry_price = 0

    final_value = cash + position * data['Close'].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    # 最大回撤
    data['Portfolio_Value'] = initial_cash
    current_cash = initial_cash
    current_position = 0
    for idx, row in data.iterrows():
        if row['Entry_Signal'] and current_position == 0:
            shares = int(current_cash / row['Close'])
            current_position = shares
            current_cash -= shares * row['Close']
        elif row['Exit_Signal'] and current_position > 0:
            current_cash += current_position * row['Close']
            current_position = 0

        data.loc[idx, 'Portfolio_Value'] = current_cash + current_position * row['Close']

    data['Cummax'] = data['Portfolio_Value'].cummax()
    data['Drawdown'] = (data['Portfolio_Value'] - data['Cummax']) / data['Cummax'] * 100
    max_drawdown = data['Drawdown'].min()

    # 胜率
    winning_trades = [t for t in trades if 'profit' in t and t['profit'] > 0]
    losing_trades = [t for t in trades if 'profit' in t and t['profit'] <= 0]
    win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0

    result = {
        'strategy': 'Bollinger Bands',
        'params': {
            'period': period,
            'std_dev': std_dev
        },
        'performance': {
            'initial_cash': round(initial_cash, 2),
            'final_value': round(final_value, 2),
            'total_return_pct': round(total_return, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'win_rate_pct': round(win_rate, 2)
        },
        'trades': trades,
        'trade_count': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades)
    }

    return result


def generate_backtest_report(result: Dict[str, Any], format: str = 'json') -> str:
    """
    生成回测报告

    Args:
        result: 回测结果字典
        format: 输出格式（'json' 或 'text'）

    Returns:
        str: 格式化后的报告
    """
    if 'error' in result:
        return f"回测失败: {result['error']}"

    if format == 'text':
        report = []
        report.append("=" * 50)
        report.append(f"策略回测报告 - {result['strategy']}")
        report.append("=" * 50)
        report.append("")

        report.append("【策略参数】")
        for key, value in result['params'].items():
            report.append(f"  {key}: {value}")
        report.append("")

        report.append("【绩效指标】")
        for key, value in result['performance'].items():
            report.append(f"  {key}: {value}")
        report.append("")

        report.append("【交易统计】")
        report.append(f"  交易次数: {result['trade_count']}")
        report.append(f"  盈利交易: {result['winning_trades']}")
        report.append(f"  亏损交易: {result['losing_trades']}")
        report.append("")

        if result['trade_count'] > 0:
            report.append("【交易记录】")
            for i, trade in enumerate(result['trades'][:10], 1):
                if 'exit_date' in trade:
                    report.append(f"  交易 {i}: {trade['date']} 买入 @ {trade['price']} -> {trade['exit_date']} 卖出 @ {trade['exit_price']}, 收益: {trade['profit']} ({trade['profit_pct']}%)")
                else:
                    report.append(f"  交易 {i}: {trade['date']} 买入 @ {trade['price']} (未平仓)")

            if len(result['trades']) > 10:
                report.append(f"  ... (共{len(result['trades'])}笔交易)")

        report.append("")
        report.append("=" * 50)

        return "\n".join(report)

    else:
        import json
        return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    print("测试回测模块...")

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'Open': [100 + i * 0.5 + np.random.randn() * 2 for i in range(100)],
        'High': [102 + i * 0.5 + np.random.randn() * 2 for i in range(100)],
        'Low': [98 + i * 0.5 + np.random.randn() * 2 for i in range(100)],
        'Close': [100 + i * 0.5 + np.random.randn() * 2 for i in range(100)],
        'Adj Close': [100 + i * 0.5 + np.random.randn() * 2 for i in range(100)],
        'Volume': [1000000 + i * 10000 for i in range(100)]
    }, index=dates)

    print(f"\n测试数据形状: {test_data.shape}")

    # 测试MA交叉策略
    print("\n测试MA交叉策略...")
    result = run_backtest(test_data, 'ma_cross', {'initial_cash': 100000})
    print(generate_backtest_report(result, 'text'))

    # 测试RSI策略
    print("\n测试RSI策略...")
    result = run_backtest(test_data, 'rsi', {'initial_cash': 100000})
    print(generate_backtest_report(result, 'text'))

    print("\n✅ 测试完成")

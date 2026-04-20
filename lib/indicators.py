"""
技术指标计算模块
使用pandas-ta库计算20+常用技术指标
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional
import warnings

# 忽略pandas-ta的警告
warnings.filterwarnings('ignore')


def calculate_indicators(df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    计算技术指标

    Args:
        df: 包含OHLCV列的DataFrame
        params: 指标参数配置字典，可选

    Returns:
        DataFrame: 原始数据 + 所有技术指标列

    支持的指标类别：
    - 趋势指标: SMA, EMA, WMA
    - 动量指标: RSI, MACD, Stochastic, ROC
    - 波动指标: Bollinger Bands, ATR, Keltner Channels
    - 成交量指标: OBV, MFI, VWAP
    - 其他: ADX, CCI, TTM Squeeze
    """
    if df.empty:
        return df

    # 复制数据避免修改原DataFrame
    data = df.copy()

    # 默认参数配置
    default_params = {
        # 移动平均
        'sma_short': 5,
        'sma_medium': 10,
        'sma_long': 20,
        'ema_short': 12,
        'ema_long': 26,

        # MACD
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,

        # RSI
        'rsi_period': 14,

        # Bollinger Bands
        'bb_length': 20,
        'bb_std': 2,

        # ATR
        'atr_period': 14,

        # Stochastic
        'stoch_k': 14,
        'stoch_d': 3,

        # ADX
        'adx_period': 14,

        # CCI
        'cci_period': 20,

        # MFI
        'mfi_period': 14,
    }

    # 合并用户参数
    if params:
        default_params.update(params)

    params = default_params

    # ==================== 趋势指标 ====================

    # 1. Simple Moving Average (SMA)
    data[f'SMA_{params["sma_short"]}'] = ta.sma(data['Close'], length=params['sma_short'])
    data[f'SMA_{params["sma_medium"]}'] = ta.sma(data['Close'], length=params['sma_medium'])
    data[f'SMA_{params["sma_long"]}'] = ta.sma(data['Close'], length=params['sma_long'])

    # 2. Exponential Moving Average (EMA)
    data[f'EMA_{params["ema_short"]}'] = ta.ema(data['Close'], length=params['ema_short'])
    data[f'EMA_{params["ema_long"]}'] = ta.ema(data['Close'], length=params['ema_long'])

    # 3. Weighted Moving Average (WMA)
    data[f'WMA_{params["sma_long"]}'] = ta.wma(data['Close'], length=params['sma_long'])

    # ==================== 动量指标 ====================

    # 4. Relative Strength Index (RSI)
    data[f'RSI_{params["rsi_period"]}'] = ta.rsi(data['Close'], length=params['rsi_period'])

    # 5. MACD (Moving Average Convergence Divergence)
    macd_result = ta.macd(
        data['Close'],
        fast=params['macd_fast'],
        slow=params['macd_slow'],
        signal=params['macd_signal']
    )
    if macd_result is not None:
        data['MACD'] = macd_result[f'MACD_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}']
        data['MACD_Signal'] = macd_result[f'MACDh_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}']
        data['MACD_Hist'] = macd_result[f'MACDs_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}']

    # 6. Stochastic Oscillator
    stoch_result = ta.stoch(
        data['High'],
        data['Low'],
        data['Close'],
        k=params['stoch_k'],
        d=params['stoch_d']
    )
    if stoch_result is not None:
        data['STOCH_K'] = stoch_result[f'STOCHk_{params["stoch_k"]}_{params["stoch_d"]}']
        data['STOCH_D'] = stoch_result[f'STOCHd_{params["stoch_k"]}_{params["stoch_d"]}']

    # 7. Rate of Change (ROC)
    data['ROC'] = ta.roc(data['Close'], length=10)

    # 8. Williams %R
    data['WILLR'] = ta.willr(data['High'], data['Low'], data['Close'], length=14)

    # ==================== 波动指标 ====================

    # 9. Bollinger Bands
    bb_result = ta.bbands(
        data['Close'],
        length=params['bb_length'],
        std=params['bb_std']
    )
    if bb_result is not None:
        data['BB_Upper'] = bb_result[f'BBU_{params["bb_length"]}_{params["bb_std"]}']
        data['BB_Middle'] = bb_result[f'BBM_{params["bb_length"]}_{params["bb_std"]}']
        data['BB_Lower'] = bb_result[f'BBL_{params["bb_length"]}_{params["bb_std"]}']
        # BB Width = (Upper - Lower) / Middle
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']

    # 10. Average True Range (ATR)
    data[f'ATR_{params["atr_period"]}'] = ta.atr(data['High'], data['Low'], data['Close'], length=params['atr_period'])

    # 11. Keltner Channels
    kc_result = ta.kc(
        data['High'],
        data['Low'],
        data['Close'],
        length=20,
        scalar=2
    )
    if kc_result is not None:
        data['KC_Upper'] = kc_result['KCUe_20_2']
        data['KC_Middle'] = kc_result['KCBe_20_2']
        data['KC_Lower'] = kc_result['KCLo_20_2']

    # ==================== 成交量指标 ====================

    # 12. On-Balance Volume (OBV)
    data['OBV'] = ta.obv(data['Close'], data['Volume'])

    # 13. Money Flow Index (MFI)
    data[f'MFI_{params["mfi_period"]}'] = ta.mfi(
        data['High'],
        data['Low'],
        data['Close'],
        data['Volume'],
        length=params['mfi_period']
    )

    # 14. Volume Moving Average
    data['VOLUME_SMA_20'] = ta.sma(data['Volume'], length=20)

    # 15. Volume Price Trend (VPT)
    data['VPT'] = ta.vpt(data['Close'], data['Volume'])

    # ==================== 其他指标 ====================

    # 16. Average Directional Index (ADX)
    adx_result = ta.adx(
        data['High'],
        data['Low'],
        data['Close'],
        length=params['adx_period']
    )
    if adx_result is not None:
        data[f'ADX_{params["adx_period"]}'] = adx_result[f'ADX_{params["adx_period"]}']
        data[f'+DI_{params["adx_period"]}'] = adx_result[f'DMP_{params["adx_period"]}']
        data[f'-DI_{params["adx_period"]}'] = adx_result[f'DMN_{params["adx_period"]}']

    # 17. Commodity Channel Index (CCI)
    data[f'CCI_{params["cci_period"]}'] = ta.cci(
        data['High'],
        data['Low'],
        data['Close'],
        length=params['cci_period']
    )

    # 18. TTM Squeeze (Simple version using Bollinger Bands and Keltner Channels)
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns and 'KC_Upper' in data.columns and 'KC_Lower' in data.columns:
        data['Squeeze'] = np.where(data['BB_Lower'] > data['KC_Lower'], 1, 0)

    # ==================== 派生指标 ====================

    # 19. Daily Return (复权收盘价涨跌幅)
    if 'Adj Close' in data.columns:
        data['Daily_Return'] = data['Adj Close'].pct_change()
    else:
        data['Daily_Return'] = data['Close'].pct_change()

    # 20. Volatility (滚动标准差，衡量波动率)
    data['Volatility'] = data['Daily_Return'].rolling(window=7).std() * np.sqrt(7)

    # 21. Price vs SMA (价格相对于均线的位置)
    data['Price_vs_SMA20'] = (data['Close'] - data[f'SMA_{params["sma_long"]}']) / data[f'SMA_{params["sma_long"]}'] * 100

    # 22. Volume Change (成交量变化)
    data['Volume_Change'] = data['Volume'].pct_change()

    # 删除包含NaN的行（保留足够数据）
    # 通常需要至少20行数据才能计算所有指标
    min_required = params['sma_long'] + 5
    if len(data) > min_required:
        data = data.iloc[min_required:].copy()

    return data


def get_indicator_list() -> list:
    """
    获取所有支持的技术指标列表

    Returns:
        list: 指标名称列表
    """
    indicators = [
        # 趋势指标
        'SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'WMA_20',

        # 动量指标
        'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'STOCH_K', 'STOCH_D', 'ROC', 'WILLR',

        # 波动指标
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
        'ATR_14', 'KC_Upper', 'KC_Middle', 'KC_Lower',

        # 成交量指标
        'OBV', 'MFI_14', 'VOLUME_SMA_20', 'VPT',

        # 其他指标
        'ADX_14', '+DI_14', '-DI_14', 'CCI_20', 'Squeeze',

        # 派生指标
        'Daily_Return', 'Volatility', 'Price_vs_SMA20', 'Volume_Change'
    ]

    return indicators


def get_indicator_category(indicator_name: str) -> str:
    """
    获取指标所属类别

    Args:
        indicator_name: 指标名称

    Returns:
        str: 指标类别（trend/momentum/volatility/volume/other）
    """
    trend = ['SMA', 'EMA', 'WMA', 'Price_vs_SMA20']
    momentum = ['RSI', 'MACD', 'STOCH', 'ROC', 'WILLR']
    volatility = ['BB', 'ATR', 'KC', 'Squeeze', 'Volatility']
    volume = ['OBV', 'MFI', 'VOLUME', 'VPT']
    other = ['ADX', 'DI', 'CCI', 'Daily_Return', 'Volume_Change']

    for prefix in trend:
        if prefix in indicator_name:
            return 'trend'
    for prefix in momentum:
        if prefix in indicator_name:
            return 'momentum'
    for prefix in volatility:
        if prefix in indicator_name:
            return 'volatility'
    for prefix in volume:
        if prefix in indicator_name:
            return 'volume'
    for prefix in other:
        if prefix in indicator_name:
            return 'other'

    return 'other'


if __name__ == '__main__':
    print("测试技术指标计算模块...")

    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'Open': [100 + i * 0.5 for i in range(100)],
        'High': [102 + i * 0.5 for i in range(100)],
        'Low': [98 + i * 0.5 for i in range(100)],
        'Close': [100 + i * 0.5 for i in range(100)],
        'Adj Close': [100 + i * 0.5 for i in range(100)],
        'Volume': [1000000 + i * 10000 for i in range(100)]
    }, index=dates)

    print(f"\n原始数据形状: {test_data.shape}")
    print(f"原始列: {test_data.columns.tolist()}")

    result = calculate_indicators(test_data)

    print(f"\n计算后数据形状: {result.shape}")
    print(f"计算后列: {result.columns.tolist()}")
    print(f"\n计算了 {len(result.columns) - len(test_data.columns)} 个指标")

    print("\n最后5行数据:")
    print(result[['Close', 'SMA_20', 'RSI_14', 'MACD', 'BB_Upper', 'BB_Lower']].tail())

    print(f"\n支持的指标数量: {len(get_indicator_list())}")
    print("指标分类示例:")
    for ind in ['SMA_20', 'RSI_14', 'BB_Upper', 'OBV', 'MACD']:
        print(f"  {ind} -> {get_indicator_category(ind)}")

    print("\n✅ 测试完成")

import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, render_template, request, jsonify
from zhipuai import ZhipuAI
from dotenv import load_dotenv
import numpy as np
import sys
import textwrap
import traceback
from yfinance.exceptions import YFRateLimitError

# 加载环境变量
load_dotenv()

# 设置代理
https_proxy = os.getenv("HTTPS_PROXY")
http_proxy = os.getenv("HTTP_PROXY")
if https_proxy:
    os.environ["HTTPS_PROXY"] = https_proxy
if http_proxy:
    os.environ["HTTP_PROXY"] = http_proxy

# 初始化Flask应用
app = Flask(__name__)

# 设置智谱AI API密钥（建议使用环境变量）
api_key = os.getenv("ZHIPUAI_API_KEY")

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=api_key)

# 获取股票数据
def fetch_stock_data(symbol, days=30):
    import time, random

    max_retries = 5
    initial_delay = 2
    backoff_factor = 2

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = initial_delay * (backoff_factor ** attempt) + random.uniform(0, attempt)
                print(f"速率限制，等待 {delay:.1f} 秒后重试 {attempt + 1}/{max_retries}…")
                time.sleep(delay)

            print(f"尝试 {attempt + 1}/{max_retries}: 获取 {symbol} 数据")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{days}d", actions=True, auto_adjust=False)

            if not data.empty:
                print(f"✅ 成功获取数据：{len(data)} 行，最新收盘价 {data['Close'].iloc[-1]}")
                return data

            info = ticker.info
            if info.get("delisted", False):
                print(f"⚠️ {symbol} 已退市")
                return pd.DataFrame()
            print("未返回数据，可能限流或其他原因，公司长名：", info.get("longName"))

        except YFRateLimitError:
            print(f"⚠️ 速率限制错误 ({attempt + 1}/{max_retries})")
        except Exception as e:
            print("Error fetching data:", e)

    print(f"⚠️ 无法获取实时数据，使用模拟数据...")
    return generate_simulated_data(symbol, days)

# 生成模拟数据
def generate_simulated_data(symbol, days):
    from datetime import datetime, timedelta
    import random

    print(f"📊 生成模拟数据: {symbol} ({days} 天)")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    base_prices = { 'AAPL': 170, 'TSLA': 180, 'GOOGL': 120, 'MSFT': 350 }
    base = base_prices.get(symbol.upper(), 100)
    data = []
    price = base
    for _ in dates:
        change = random.uniform(-0.04, 0.04)
        price *= (1 + change)
        o = price * random.uniform(0.99, 1.01)
        h = price * random.uniform(1.002, 1.025)
        l = price * random.uniform(0.975, 0.998)
        data.append({'Open': round(o,2),'High': round(h,2),'Low': round(l,2),'Close': round(price,2),'Adj Close': round(price,2),'Volume': random.randint(1e7,8e7)})
    df = pd.DataFrame(data, index=dates)
    print(f"✅ 模拟数据生成完毕，共 {len(df)} 行")
    return df

# 使用智谱AI分析股票数据
def analyze_stock_with_ai(stock_data, symbol, has_position=False, cost_price=None, shares=None):
    # 预处理
    data = stock_data.copy()

    # 检查是否包含'Adj Close'列
    if 'Adj Close' not in data.columns:
        if 'Close' in data.columns:
            print("⚠️ 数据中缺少'Adj Close'列，使用'Close'列代替")
            data['Adj Close'] = data['Close']
        else:
            error_msg = "🚨 数据中缺少'Adj Close'和'Close'列，无法进行分析"
            print(error_msg)
            return error_msg

    # 基础技术指标
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=7).std() * np.sqrt(7)
    data['MA5'] = data['Adj Close'].rolling(window=5).mean()
    data['MA20'] = data['Adj Close'].rolling(window=20).mean()

    # 成交量相关指标
    if 'Volume' in data.columns:
        data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
        data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
        # 量价配合：价格涨跌与量能对比
        data['Price_Up'] = data['Adj Close'].diff() > 0
        data['Volume_Surge'] = data['Volume'] > (data['Volume_MA20'] * 1.5)
        data['Up_With_Surge'] = data['Price_Up'] & data['Volume_Surge']
        data['Down_With_Surge'] = (~data['Price_Up']) & data['Volume_Surge']

    # 波动率与ATR（近似计算）
    if all(col in data.columns for col in ['High', 'Low', 'Adj Close']):
        tr1 = data['High'] - data['Low']
        tr2 = (data['High'] - data['Adj Close'].shift(1)).abs()
        tr3 = (data['Low'] - data['Adj Close'].shift(1)).abs()
        data['TrueRange'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['ATR14'] = data['TrueRange'].rolling(window=14).mean()

    # MACD
    ema12 = data['Adj Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Adj Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal']

    # RSI(14)
    delta = data['Adj Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    data['RSI14'] = 100 - (100 / (1 + rs))

    # 确保用于显示的数据列存在
    display_cols = ['Open','High','Low','Adj Close','Volume']
    available_cols = [col for col in display_cols if col in data.columns]
    recent = data.tail(10)[available_cols].to_string()

    # 汇总指标统计
    describe_cols = [c for c in ['Daily Return','Volatility','MA5','MA20','Volume','Volume_MA5','Volume_MA20','ATR14','MACD','Signal','MACD_Hist','RSI14'] if c in data.columns]
    indicators = data[describe_cols].describe().to_string()

    # 获取当前价格
    current_price = data['Adj Close'].iloc[-1]
    prev_price = data['Adj Close'].iloc[-2] if len(data) > 1 else current_price
    change_pct_last = ((current_price - prev_price) / prev_price * 100) if prev_price else 0

    # 快照：量能、动量、波动
    volume_last = data['Volume'].iloc[-1] if 'Volume' in data.columns else None
    volume_ma20_last = data['Volume_MA20'].iloc[-1] if 'Volume_MA20' in data.columns else None
    volume_ratio = (volume_last / volume_ma20_last) if (volume_last and volume_ma20_last and volume_ma20_last > 0) else None
    rsi_last = data['RSI14'].iloc[-1] if 'RSI14' in data.columns else None
    macd_last = data['MACD'].iloc[-1] if 'MACD' in data.columns else None
    signal_last = data['Signal'].iloc[-1] if 'Signal' in data.columns else None
    atr_last = data['ATR14'].iloc[-1] if 'ATR14' in data.columns else None

    # 公司行为（最近非零）
    corp_actions = []
    if 'Dividends' in data.columns:
        recent_div = data['Dividends'][data['Dividends'] != 0.0].tail(1)
        if len(recent_div) > 0:
            corp_actions.append(f"最近分红: {recent_div.index[-1].date()} 金额 {recent_div.iloc[-1]}")
    if 'Stock Splits' in data.columns:
        recent_split = data['Stock Splits'][data['Stock Splits'] != 0.0].tail(1)
        if len(recent_split) > 0:
            corp_actions.append(f"最近拆股: {recent_split.index[-1].date()} 比例 {recent_split.iloc[-1]}")
    corp_actions_text = "\n    ".join(corp_actions) if corp_actions else "无显著公司行为"

    # 安全格式化函数
    def format_optional(val, digits=None, int_like=False):
        try:
            if val is None or (isinstance(val, float) and (np.isnan(val))):
                return "N/A"
        except Exception:
            pass
        if int_like:
            try:
                return f"{int(val)}"
            except Exception:
                return str(val)
        if digits is not None:
            try:
                return f"{float(val):.{digits}f}"
            except Exception:
                return str(val)
        return str(val)

    current_price_text = format_optional(current_price, 2)
    prev_price_text = format_optional(prev_price, 2)
    change_pct_last_text = format_optional(change_pct_last, 2)
    volume_last_text = format_optional(volume_last, None, int_like=True)
    volume_ma20_text = format_optional(volume_ma20_last, None, int_like=True)
    volume_ratio_text = format_optional(volume_ratio, 2)
    rsi_text = format_optional(rsi_last, 2)
    macd_text = format_optional(macd_last, 4)
    signal_text = format_optional(signal_last, 4)
    atr_text = format_optional(atr_last, 4)

    # 构建用户持仓信息
    position_info = ""
    if has_position and cost_price and shares:
        profit_loss = (current_price - cost_price) * shares
        profit_loss_pct = ((current_price - cost_price) / cost_price) * 100
        position_info = f"""

    用户持仓信息:
    - 持仓成本: {cost_price} 元/股
    - 持股数量: {shares} 股
    - 当前价格: {current_price_text} 元/股
    - 盈亏金额: {profit_loss:.2f} 元
    - 盈亏比例: {profit_loss_pct:.2f}%
        """

    prompt = textwrap.dedent(f"""
    作为一名专业股票分析师，请分析以下 {symbol} 股票数据并提供投资建议。

    数据周期: {len(data)} 天

    最近 10 天交易数据:
    {recent}

    技术指标统计:
    {indicators}
    {position_info}

    关键快照:
    - 当前价格: {current_price_text} 元/股，昨日收盘: {prev_price_text}，当日涨跌幅: {change_pct_last_text}%
    - 成交量: {volume_last_text}，20日均量: {volume_ma20_text}，量能比: {volume_ratio_text}
    - RSI14: {rsi_text}
    - MACD: {macd_text}，Signal: {signal_text}
    - ATR14: {atr_text}
    - 公司行为: {corp_actions_text}

    可用指标说明:
    - 均线: MA5/MA20
    - 成交量均线: Volume_MA5/Volume_MA20；放量阈值约为20日均量的1.5倍
    - 量价关系: Up_With_Surge(价涨放量)/Down_With_Surge(价跌放量)
    - 波动率: 7日标准差换算；ATR14 反映真实波动幅度
    - 动量: MACD/Signal/MACD_Hist
    - 超买超卖: RSI14（>70 可能过热，<30 可能超卖）

    分析要点:
    1. 价格趋势与关键支撑/阻力位
    2. 成交量变化与量价配合（放量上涨/放量下跌、缩量回调等）
    3. 波动性与风险（ATR14/Volatility）
    4. 移动平均线交叉与背离（MA、MACD）
    {"5. 基于当前持仓的盈亏与仓位管理建议" if has_position else ""}

    请提供:
    {"- 明确的买入/持有/卖出建议（考虑当前持仓盈亏情况）" if has_position else "- 明确的买入建议（是否适合建仓）"}
    - 建议仓位大小 (如适用)
    - 关键入/出场点位
    - 风险提示与止损建议（可参考ATR14或近期波动）
    {"- 针对当前持仓的具体操作建议" if has_position else ""}

    分析需简洁专业，尽量引用上述指标进行客观判断。

    重要要求：请在最后单独一行输出：
    决策: {"买入/卖出/持有" if has_position else "买入/观望"}
    只允许输出上述枚举值之一，不要添加其它词或标点。
    """ )

    try:
        print("开始调用智谱AI API...")
        print(f"API密钥状态: {'已设置' if api_key else '未设置'}")
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role":"user","content":prompt}]
        )
        print("智谱AI API调用成功")
        return response.choices[0].message.content
    except TimeoutError:
        error_msg = "🚨 AI 分析超时错误"
        print(f"{error_msg}: {traceback.format_exc()}")
        return error_msg
    except ConnectionError:
        error_msg = "🚨 AI 分析连接错误"
        print(f"{error_msg}: {traceback.format_exc()}")
        return error_msg
    except Exception as e:
        error_msg = f"🚨 AI 分析出错: {str(e)}"
        print(f"{error_msg}: {traceback.format_exc()}")
        return error_msg

# Flask 路由
@app.route('/')
def index():
    return render_template('index.html')
# 关于页面路由
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        symbol = request.json.get('symbol', 'AAPL')
        days = int(request.json.get('days', 30))
        has_position = request.json.get('has_position', False)
        cost_price = request.json.get('cost_price')
        shares = request.json.get('shares')

        df = fetch_stock_data(symbol, days)
        if df.empty:
            return jsonify({"error": "未获取到数据"}), 404

        # 如果用户选择已持股，验证必要参数
        if has_position:
            if not cost_price or not shares:
                return jsonify({"error": "已持股状态下需要提供持仓成本和持股数量"}), 400
            try:
                cost_price = float(cost_price)
                shares = int(shares)
                if cost_price <= 0 or shares <= 0:
                    return jsonify({"error": "持仓成本和持股数量必须大于0"}), 400
            except (ValueError, TypeError):
                return jsonify({"error": "持仓成本和持股数量必须是有效数字"}), 400

        result = analyze_stock_with_ai(df, symbol, has_position, cost_price, shares)

        # 准备图表数据
        dates = df.index.strftime('%Y-%m-%d').tolist()
        if 'Adj Close' in df.columns:
            prices = df['Adj Close'].tolist()
        elif 'Close' in df.columns:
            prices = df['Close'].tolist()
        else:
            prices = []

        return jsonify({
            "symbol": symbol,
            "analysis": result,
            "dates": dates,
            "prices": prices,
            "has_position": has_position,
            "cost_price": cost_price,
            "shares": shares
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

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
            data = ticker.history(period=f"{days}d")

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
def analyze_stock_with_ai(stock_data, symbol):
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

    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=7).std() * np.sqrt(7)
    data['MA5'] = data['Adj Close'].rolling(window=5).mean()
    data['MA20'] = data['Adj Close'].rolling(window=20).mean()

    # 确保用于显示的数据列存在
    display_cols = ['Open','High','Low','Adj Close','Volume']
    available_cols = [col for col in display_cols if col in data.columns]
    recent = data.tail(10)[available_cols].to_string()

    indicators = data[['Daily Return','Volatility','MA5','MA20']].describe().to_string()

    prompt = textwrap.dedent(f"""
    作为一名专业股票分析师，请分析以下 {symbol} 股票数据并提供投资建议。

    数据周期: {len(data)} 天

    最近 10 天交易数据:
    {recent}

    技术指标统计:
    {indicators}

    分析要点:
    1. 价格趋势与关键支撑/阻力位
    2. 交易量变化与市场情绪
    3. 波动性分析与风险评估
    4. 移动平均线交叉信号

    请提供:
    - 明确的买入/持有/卖出建议
    - 建议仓位大小 (如适用)
    - 关键入/出场点位
    - 风险提示与止损建议

    分析需简洁专业，基于提供的数据客观判断。
    """)

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
        df = fetch_stock_data(symbol, days)
        if df.empty:
            return jsonify({"error": "未获取到数据"}), 404
        result = analyze_stock_with_ai(df, symbol)

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
            "prices": prices
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

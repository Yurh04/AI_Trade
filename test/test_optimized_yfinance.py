import yfinance as yf
import time, random
from yfinance.exceptions import YFRateLimitError

def test_optimized_fetch(symbol="AAPL", days=30):
    print(f"测试优化后的 yfinance 连接 - 股票代码: {symbol}, 周期: {days} 天")

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
            ticker = yf.Ticker(symbol)  # 不再传 session
            data = ticker.history(period=f"{days}d")

            if not data.empty:
                print(f"✅ 成功获取数据：{len(data)} 行，最新收盘价 {data['Close'].iloc[-1]}")
                return True
            else:
                info = ticker.info
                if info.get("delisted", False):
                    print(f"⚠️ {symbol} 已退市")
                else:
                    print("未返回数据，可能被限流或其他原因，company 长名：", info.get("longName"))
        except YFRateLimitError:
            print(f"⚠️ 速率限制错误 ({attempt + 1}/{max_retries})")
        except Exception as e:
            print("错误:", e)
    print("❌ 所有重试失败")
    return False

if __name__ == "__main__":
    test_optimized_fetch()

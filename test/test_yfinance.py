import yfinance as yf
import time

print("Testing yfinance connection...")

# 测试1: 最简单的方法
try:
    print("Test 1: Simple download")
    data = yf.download("AAPL", period="5d", progress=False)
    if not data.empty:
        print(f"✅ Success: Got {len(data)} days of data")
        print(data.tail(2))
    else:
        print("❌ No data returned")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50 + "\n")

# 测试2: 使用Ticker对象
try:
    print("Test 2: Ticker object")
    ticker = yf.Ticker("AAPL")
    info = ticker.info
    print(f"✅ Company name: {info.get('longName', 'Unknown')}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50 + "\n")

# 测试3: 获取历史数据
try:
    print("Test 3: History data")
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="3d")
    if not data.empty:
        print(f"✅ Success: Got {len(data)} days of history")
        print(data.tail(1))
    else:
        print("❌ No history data")
except Exception as e:
    print(f"❌ Error: {e}")

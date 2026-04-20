# A股数据获取模块使用说明

## 概述

`lib/data_fetcher.py` 模块提供了A股日线数据获取功能，使用Akshare库作为数据源。

## 功能特性

- ✅ 使用Akshare免费数据源
- ✅ 支持A股股票代码格式（000001.SZ, 600000.SH）
- ✅ 返回格式与yfinance一致的DataFrame
- ✅ 包含错误处理和重试机制（5次重试，指数退避）
- ✅ 提供模拟数据生成功能（用于测试或API不可用时）
- ✅ 中文注释和emoji状态指示

## 安装依赖

```bash
pip install akshare==1.18.32
```

或更新requirements.txt后安装：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 导入模块

```python
from lib.data_fetcher import fetch_cn_stock_data, generate_simulated_cn_data
```

### 2. 获取A股数据

```python
# 获取平安银行（000001.SZ）最近30天的数据
df = fetch_cn_stock_data("000001.SZ", days=30)

# 获取浦发银行（600000.SH）最近60天的数据
df = fetch_cn_stock_data("600000.SH", days=60)
```

### 3. 生成模拟数据

```python
# 生成模拟数据（用于测试或API不可用时）
df = generate_simulated_cn_data("000001.SZ", days=30)
```

## 返回数据格式

返回的DataFrame包含以下列：

| 列名 | 说明 |
|------|------|
| Open | 开盘价 |
| High | 最高价 |
| Low | 最低价 |
| Close | 收盘价 |
| Adj Close | 复权收盘价（前复权） |
| Volume | 成交量 |

索引为DatetimeIndex，格式为日期。

## 示例

```python
from lib.data_fetcher import fetch_cn_stock_data

# 获取数据
df = fetch_cn_stock_data("000001.SZ", days=30)

# 查看数据
print(df.head())
print(f"数据形状: {df.shape}")
print(f"最新收盘价: {df['Close'].iloc[-1]}")
```

## 错误处理

模块内置了错误处理和重试机制：

- 最多重试5次
- 使用指数退避策略（2秒、4秒、8秒、16秒、32秒）
- 网络错误时自动重试
- 所有重试失败后返回空DataFrame

## 注意事项

1. **网络要求**：需要能够访问东方财富API（push2his.eastmoney.com）
2. **数据延迟**：Akshare数据可能有1-2天的延迟
3. **交易日**：返回的数据只包含交易日（排除周末和节假日）
4. **股票代码**：支持带后缀的格式（000001.SZ, 600000.SH），会自动转换为Akshare需要的格式

## 测试

运行测试脚本：

```bash
python test/test_data_fetcher.py
```

或直接运行模块：

```bash
python lib/data_fetcher.py
```

## 与app.py集成

可以在app.py中集成A股数据获取功能：

```python
from lib.data_fetcher import fetch_cn_stock_data

# 在Flask路由中使用
@app.route('/analyze-cn', methods=['POST'])
def analyze_cn():
    try:
        symbol = request.json.get('symbol', '000001.SZ')
        days = int(request.json.get('days', 30))
        df = fetch_cn_stock_data(symbol, days)

        if df.empty:
            return jsonify({"error": "未获取到数据"}), 404

        # 使用AI分析数据
        result = analyze_stock_with_ai(df, symbol)

        return jsonify({
            "symbol": symbol,
            "analysis": result,
            "dates": df.index.strftime('%Y-%m-%d').tolist(),
            "prices": df['Close'].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

## 参考资源

- Akshare官方文档: https://akshare.akfamily.xyz/
- 东方财富API: https://push2his.eastmoney.com/

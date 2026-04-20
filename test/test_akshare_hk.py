"""
测试Akshare港股数据获取
"""
import akshare as ak
from datetime import datetime, timedelta

print("=" * 60)
print("测试Akshare港股数据获取")
print("=" * 60)

# 测试腾讯控股（00700）
print("\n=== 测试 00700（腾讯控股）===")
try:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)  # 只取10天数据以加快测试

    print(f"开始日期: {start_date.strftime('%Y%m%d')}")
    print(f"结束日期: {end_date.strftime('%Y%m%d')}")

    data = ak.stock_hk_hist(
        symbol="00700",
        period="daily",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust="qfq"
    )

    if data is not None and not data.empty:
        print(f"✅ 成功获取数据：{len(data)} 行")
        print(f"✅ 列名: {data.columns.tolist()}")
        print(f"✅ 前5行:\n{data.head()}")
    else:
        print("⚠️ 未获取到数据")

except Exception as e:
    print(f"🚨 错误: {str(e)}")
    import traceback
    print(f"错误详情: {traceback.format_exc()}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

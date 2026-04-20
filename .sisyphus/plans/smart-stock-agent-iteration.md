# 智能股票分析Agent迭代方案

## TL;DR

> **Quick Summary**: 将基础股票分析应用升级为智能量化策略开发平台，支持美股/A股/港股多市场，集成回测框架、高级可视化、自然语言交互和持续学习能力，1-2周完成核心功能。
>
> **Deliverables**:
> - 多市场数据获取（美股/A股/港股）
> - 丰富技术指标库（20+指标）
> - 策略回测框架（vectorbt集成）
> - 高级可视化界面（Plotly Dash）
> - 自然语言股票查询
> - AI分析记忆和个性化
> - 自动交易信号生成
>
> **Estimated Effort**: Medium-Large (1-2周)
> **Parallel Execution**: YES - 5 waves
> **Critical Path**: 数据层 → 技术指标 → 回测框架 → AI增强 → 可视化集成

---

## Context

### Original Request
"希望这个股票trade agent更加智能，具有实用价值，给我迭代方案"

### Interview Summary
**Key Discussions**:

1. **智能化方向（全选）**:
   - 分析更准确深入：多维度数据整合，技术面+基本面+情绪面
   - 自动化决策：自动生成交易信号、仓位建议、止损止盈点
   - 自然交互：自然语言对话查询股票、追问分析细节
   - 持续学习优化：记住历史分析、学习用户偏好

2. **使用场景**: 量化策略开发
   - 需要策略回测功能
   - 需要丰富的技术指标
   - 需要策略验证能力

3. **技术约束**:
   - 市场范围：美股 + A股 + 港股
   - 数据频率：日线数据即可
   - 时间预算：1-2周快速迭代
   - 数据源：优先免费方案（Akshare）
   - 交易执行：仅分析建议，不自动执行
   - 用户规模：单用户
   - 可视化：高级可视化（Plotly/TradingView）
   - 测试：保持手动测试

**Research Findings**:
- 当前项目是单文件Flask应用（app.py）
- 技术指标非常基础（仅MA5/MA20/Volatility/Daily Return）
- 无专业量化库集成
- 无回测功能
- 无数据缓存机制
- 无历史分析记忆

### Technical Decisions

**Data Layer**:
- 美股: 保持yfinance
- A股/港股: Akshare（免费、易用、数据全面）
- 缓存: SQLite本地数据库

**Analysis Layer**:
- 技术指标: pandas-ta（比TA-Lib易安装，功能丰富）
- 回测框架: vectorbt（高性能，支持向量化回测）
- AI分析: ZhipuAI（现有）+ prompt工程优化

**Interaction Layer**:
- 自然语言: 利用ZhipuAI对话能力
- 可视化: Plotly Dash（交互式图表，易集成）
- UI: Flask + 现有HTML模板升级

**Intelligence Layer**:
- 策略记忆: SQLite存储历史分析结果
- 个性化: 记录用户反馈，调整分析权重
- 多维度分析: 技术面+基本面+情绪面综合评分

---

## Work Objectives

### Core Objective
构建一个智能量化策略开发平台，具备多市场数据获取、策略回测、高级可视化、自然语言交互和持续学习能力，支撑个人量化策略开发和研究。

### Concrete Deliverables
- `lib/data_fetcher.py` - 多市场数据获取模块（美股/A股/港股）
- `lib/indicators.py` - 技术指标计算模块（20+指标）
- `lib/backtest.py` - 策略回测模块（vectorbt集成）
- `lib/ai_memory.py` - AI记忆和个性化模块
- `lib/signal_generator.py` - 交易信号生成模块
- `app.py` - 升级后的主应用（集成所有模块）
- `templates/dashboard.html` - 高级可视化仪表板
- `data/analysis_memory.db` - SQLite历史分析数据库

### Definition of Done
- [ ] 可以同时获取美股、A股、港股数据
- [ ] 可以计算20+种技术指标
- [ ] 可以运行策略回测并生成报告
- [ ] 可以通过自然语言查询股票
- [ ] AI分析包含历史记忆和个性化建议
- [ ] 可视化图表展示回测结果和技术指标
- [ ] 自动生成交易信号和仓位建议
- [ ] 手动测试脚本验证所有功能

### Must Have
- 多市场数据获取（美股/A股/港股）
- 技术指标计算（至少15种常用指标）
- 策略回测功能（vectorbt集成）
- 自然语言查询股票
- AI记忆能力（记住历史分析）
- 高级可视化（Plotly图表）
- 自动交易信号生成

### Must NOT Have (Guardrails)
- ❌ 不重构为多模块架构（保持单文件Flask为主）
- ❌ 不引入付费数据源
- ❌ 不实现实盘交易执行
- ❌ 不引入复杂的多用户认证系统
- ❌ 不实现实时tick数据推送
- ❌ 不支持高频交易策略
- ❌ 不引入pytest自动化测试框架
- ❌ 不过度设计（保持MVP思路）

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES（手动测试脚本）
- **Automated tests**: None（保持手动测试）
- **Framework**: 无（继续使用print-based验证）
- **Test approach**: 手动测试脚本 + 实际运行验证

### QA Policy
每个任务包含Agent-Executed QA Scenarios，验证实际功能运行。

- **API验证**: Bash (curl) - 测试Flask API端点
- **数据验证**: Python脚本 - 验证数据获取和指标计算
- **回测验证**: Python脚本 - 运行简单回测策略
- **可视化验证**: 手动检查图表渲染

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - 数据层和基础架构):
├── Task 1: SQLite数据库设计和初始化 [quick]
├── Task 2: A股数据获取模块（Akshare集成）[unspecified-high]
├── Task 3: 港股数据获取模块（Akshare集成）[unspecified-high]
├── Task 4: 数据缓存机制实现 [quick]
└── Task 5: 统一数据接口设计 [quick]

Wave 2 (Analysis Layer - 技术指标和回测):
├── Task 6: pandas-ta集成和指标计算模块 [unspecified-high]
├── Task 7: 技术指标API端点 [quick]
├── Task 8: vectorbt回测框架集成 [deep]
├── Task 9: 回测结果生成和格式化 [unspecified-high]
└── Task 10: 策略参数配置接口 [quick]

Wave 3 (Intelligence Layer - AI增强):
├── Task 11: AI记忆模块设计和实现 [unspecified-high]
├── Task 12: 个性化权重调整机制 [quick]
├── Task 13: 多维度分析整合（技术+基本面+情绪）[deep]
├── Task 14: 自然语言查询接口 [unspecified-high]
└── Task 15: Prompt工程优化 [quick]

Wave 4 (Decision Layer - 交易信号):
├── Task 16: 交易信号生成算法 [deep]
├── Task 17: 仓位建议计算 [quick]
├── Task 18: 止损止盈点计算 [quick]
└── Task 19: 风险评估模块 [unspecified-high]

Wave 5 (Presentation Layer - 可视化):
├── Task 20: Plotly Dash集成 [unspecified-high]
├── Task 21: 技术指标可视化 [visual-engineering]
├── Task 22: 回测结果可视化 [visual-engineering]
├── Task 23: 交互式仪表板设计 [visual-engineering]
└── Task 24: 主界面集成和优化 [quick]

Wave FINAL (Verification - 最终验证):
├── Task F1: 多市场数据获取验证 [unspecified-high]
├── Task F2: 回测功能完整性验证 [deep]
├── Task F3: 自然语言交互验证 [unspecified-high]
└── Task F4: 整体功能集成测试 [deep]

Critical Path: T1-T5 → T6-T10 → T11-T15 → T16-T19 → T20-T24 → F1-F4
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 5 (Waves 1-4)
```

### Dependency Matrix

- **1-5**: — — 6-15, 1
- **6-7**: 5 — 8-10, 21, 2
- **8-10**: 6-7 — 16-19, 22, 3
- **11-15**: 5, 6-7 — 16-19, 4
- **16-19**: 6-10, 11-15 — F1-F4, 5
- **20-24**: 6-10 — F1-F4, 6
- **F1-F4**: 16-19, 20-24 — —

---

## TODOs

### Wave 1: Foundation - 数据层和基础架构

- [ ] 1. SQLite数据库设计和初始化

  **What to do**:
  - 创建SQLite数据库文件 `data/analysis_memory.db`
  - 设计表结构：股票数据缓存表、历史分析表、用户偏好表
  - 实现数据库初始化脚本
  - 添加数据库连接管理

  **Must NOT do**:
  - 不要使用复杂的ORM（保持简单，使用sqlite3）
  - 不要设计过于复杂的表结构（MVP思路）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 数据库设计和初始化是基础工作，相对简单
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `git-master`: 不涉及复杂git操作

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: Tasks 4, 11 (依赖数据库)
  - **Blocked By**: None

  **References**:
  - `app.py:21-24` - 环境变量和配置模式参考
  - Python sqlite3官方文档: https://docs.python.org/3/library/sqlite3.html

  **Acceptance Criteria**:
  - [ ] data/analysis_memory.db文件创建成功
  - [ ] 包含stock_cache、analysis_history、user_preferences三张表
  - [ ] 数据库连接和关闭正常工作

  **QA Scenarios**:
  ```
  Scenario: 数据库初始化验证
    Tool: Python脚本
    Steps:
      1. 运行数据库初始化脚本
      2. 查询sqlite_master表验证表结构
      3. 插入测试数据并读取
    Expected Result: 表结构正确，CRUD操作正常
    Evidence: .sisyphus/evidence/task-1-db-init.txt
  ```

  **Commit**: YES
  - Message: `feat(data): add SQLite database initialization`
  - Files: `data/analysis_memory.db`, `lib/db.py`

- [ ] 2. A股数据获取模块（Akshare集成）

  **What to do**:
  - 安装akshare库到requirements.txt
  - 实现A股数据获取函数 `fetch_cn_stock_data(symbol, days)`
  - 处理A股股票代码格式（如000001.SZ）
  - 实现错误处理和重试机制

  **Must NOT do**:
  - 不要引入付费数据源
  - 不要获取实时tick数据

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要学习Akshare API并集成，工作量较大
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5)
  - **Blocks**: Task 5 (统一数据接口)
  - **Blocked By**: None

  **References**:
  - `app.py:27-61` - yfinance数据获取模式参考
  - Akshare官方文档: https://akshare.akfamily.xyz/

  **Acceptance Criteria**:
  - [ ] requirements.txt添加akshare
  - [ ] 可以获取A股日线数据
  - [ ] 返回数据格式与美股数据一致
  - [ ] 包含错误处理和日志

  **QA Scenarios**:
  ```
  Scenario: 获取A股数据
    Tool: Python脚本
    Steps:
      1. 调用fetch_cn_stock_data("000001.SZ", 30)
      2. 验证返回DataFrame包含OHLCV字段
      3. 验证数据行数 >= 30
    Expected Result: 成功获取A股数据，格式正确
    Evidence: .sisyphus/evidence/task-2-cn-data.txt
  ```

  **Commit**: YES
  - Message: `feat(data): add A-share data fetching with Akshare`

- [ ] 3. 港股数据获取模块（Akshare集成）

  **What to do**:
  - 实现港股数据获取函数 `fetch_hk_stock_data(symbol, days)`
  - 处理港股股票代码格式（如00700.HK）
  - 实现错误处理和重试机制

  **Must NOT do**:
  - 不要引入付费数据源

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要学习Akshare港股API并集成
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:
  - `app.py:27-61` - 数据获取模式参考
  - Akshare港股文档

  **Acceptance Criteria**:
  - [ ] 可以获取港股日线数据
  - [ ] 返回数据格式统一
  - [ ] 包含错误处理

  **QA Scenarios**:
  ```
  Scenario: 获取港股数据
    Tool: Python脚本
    Steps:
      1. 调用fetch_hk_stock_data("00700.HK", 30)
      2. 验证返回DataFrame格式正确
      3. 验证数据来源
    Expected Result: 成功获取港股数据
    Evidence: .sisyphus/evidence/task-3-hk-data.txt
  ```

  **Commit**: YES
  - Message: `feat(data): add HK stock data fetching`

- [ ] 4. 数据缓存机制实现

  **What to do**:
  - 实现数据缓存到SQLite的逻辑
  - 设置缓存过期时间（如1小时）
  - 实现缓存查询和更新函数
  - 缓存键包含：symbol + market + days

  **Must NOT do**:
  - 不要缓存过久（避免数据过时）
  - 不要缓存敏感数据

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 缓存逻辑相对简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1 (依赖数据库)

  **References**:
  - `app.py:27-61` - 数据获取流程参考

  **Acceptance Criteria**:
  - [ ] 数据可以缓存到SQLite
  - [ ] 缓存过期时间可配置
  - [ ] 缓存命中时直接返回数据

  **QA Scenarios**:
  ```
  Scenario: 缓存命中验证
    Tool: Python脚本
    Steps:
      1. 首次获取数据，记录时间
      2. 第二次获取，验证从缓存读取
      3. 验证缓存时间戳
    Expected Result: 缓存机制正常工作
    Evidence: .sisyphus/evidence/task-4-cache.txt
  ```

  **Commit**: YES
  - Message: `feat(data): add data caching mechanism`

- [ ] 5. 统一数据接口设计

  **What to do**:
  - 创建统一数据获取函数 `fetch_stock_data_unified(symbol, market, days)`
  - market参数支持: "US", "CN", "HK"
  - 统一返回数据格式
  - 自动选择数据源并应用缓存

  **Must NOT do**:
  - 不要破坏现有API兼容性

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 接口整合工作相对直接
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4)
  - **Blocks**: Tasks 6-24
  - **Blocked By**: Tasks 2, 3, 4

  **References**:
  - `app.py:27-61` - 现有数据获取函数

  **Acceptance Criteria**:
  - [ ] 统一接口可以获取三市场数据
  - [ ] 数据格式完全统一
  - [ ] 缓存机制集成

  **QA Scenarios**:
  ```
  Scenario: 多市场数据获取
    Tool: Python脚本
    Steps:
      1. 获取美股AAPL
      2. 获取A股000001.SZ
      3. 获取港股00700.HK
      4. 验证数据格式一致性
    Expected Result: 三市场数据格式统一
    Evidence: .sisyphus/evidence/task-5-unified.txt
  ```

  **Commit**: YES
  - Message: `feat(data): add unified stock data interface`

---

### Wave 2: Analysis Layer - 技术指标和回测

- [ ] 6. pandas-ta集成和指标计算模块

  **What to do**:
  - 安装pandas-ta到requirements.txt
  - 实现技术指标计算函数 `calculate_indicators(df)`
  - 支持15+常用指标：RSI, MACD, Bollinger Bands, ATR, KDJ, OBV等
  - 实现指标参数可配置

  **Must NOT do**:
  - 不要引入TA-Lib（安装复杂）
  - 不要计算过于冷门的指标

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要学习pandas-ta并实现多个指标计算
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 9, 10)
  - **Blocks**: Tasks 7, 8, 21
  - **Blocked By**: Task 5

  **References**:
  - `app.py:89-107` - 现有指标计算模式
  - pandas-ta文档: https://github.com/twopirllc/pandas-ta

  **Acceptance Criteria**:
  - [ ] pandas-ta成功安装
  - [ ] 可以计算至少15种技术指标
  - [ ] 指标参数可配置
  - [ ] 返回包含所有指标的DataFrame

  **QA Scenarios**:
  ```
  Scenario: 技术指标计算验证
    Tool: Python脚本
    Steps:
      1. 获取AAPL数据
      2. 调用calculate_indicators
      3. 验证包含RSI, MACD, BB等指标
      4. 验证指标值合理性
    Expected Result: 成功计算15+指标，无NaN错误
    Evidence: .sisyphus/evidence/task-6-indicators.txt
  ```

  **Commit**: YES
  - Message: `feat(analysis): add technical indicators with pandas-ta`

- [ ] 7. 技术指标API端点

  **What to do**:
  - 添加Flask路由 `/api/indicators`
  - 接收symbol, market参数
  - 返回技术指标JSON数据
  - 包含错误处理

  **Must NOT do**:
  - 不要返回过多数据（考虑分页）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: API端点实现简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 8, 9, 10)
  - **Blocks**: Task 21
  - **Blocked By**: Task 6

  **References**:
  - `app.py:171-197` - 现有API端点模式

  **Acceptance Criteria**:
  - [ ] API端点可访问
  - [ ] 返回JSON格式指标数据
  - [ ] 包含错误处理

  **QA Scenarios**:
  ```
  Scenario: API指标查询
    Tool: Bash (curl)
    Steps:
      1. curl POST /api/indicators -d '{"symbol":"AAPL","market":"US"}'
      2. 验证返回JSON包含指标
      3. 验证status code 200
    Expected Result: API返回正确指标数据
    Evidence: .sisyphus/evidence/task-7-api.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add indicators endpoint`

- [ ] 8. vectorbt回测框架集成

  **What to do**:
  - 安装vectorbt到requirements.txt
  - 实现回测引擎函数 `run_backtest(data, strategy, params)`
  - 支持简单策略：MA交叉、RSI超买超卖
  - 生成回测报告：收益率、最大回撤、夏普比率

  **Must NOT do**:
  - 不要实现过于复杂的策略
  - 不要使用实时数据回测

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 回测框架集成和策略实现较复杂
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 9, 10)
  - **Blocks**: Tasks 9, 16, 22
  - **Blocked By**: Task 6

  **References**:
  - vectorbt文档: https://vectorbt.dev/

  **Acceptance Criteria**:
  - [ ] vectorbt成功安装
  - [ ] 可以运行MA交叉策略回测
  - [ ] 返回完整回测报告
  - [ ] 回测速度合理（<10秒）

  **QA Scenarios**:
  ```
  Scenario: MA交叉策略回测
    Tool: Python脚本
    Steps:
      1. 准备AAPL数据
      2. 定义MA5上穿MA20买入策略
      3. 运行回测，初始资金100000
      4. 验证返回报告包含收益率等指标
    Expected Result: 回测成功，报告完整
    Evidence: .sisyphus/evidence/task-8-backtest.txt
  ```

  **Commit**: YES
  - Message: `feat(backtest): add vectorbt backtesting framework`

- [ ] 9. 回测结果生成和格式化

  **What to do**:
  - 实现回测报告生成函数 `generate_backtest_report(result)`
  - 格式化关键指标：年化收益、最大回撤、夏普比率、胜率
  - 生成交易记录详情
  - 支持JSON和文本两种输出格式

  **Must NOT do**:
  - 不要生成过于冗长的报告

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 报告格式化和数据提取需要工作量
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 8, 10)
  - **Blocks**: Task 22
  - **Blocked By**: Task 8

  **References**:
  - vectorbt回测结果API

  **Acceptance Criteria**:
  - [ ] 报告包含所有关键指标
  - [ ] 交易记录清晰
  - [ ] 支持JSON格式输出

  **QA Scenarios**:
  ```
  Scenario: 回测报告生成
    Tool: Python脚本
    Steps:
      1. 运行简单回测
      2. 调用generate_backtest_report
      3. 验证JSON格式正确
      4. 验证包含所有必需字段
    Expected Result: 报告格式正确，内容完整
    Evidence: .sisyphus/evidence/task-9-report.txt
  ```

  **Commit**: YES
  - Message: `feat(backtest): add backtest report generation`

- [ ] 10. 策略参数配置接口

  **What to do**:
  - 添加Flask路由 `/api/backtest`
  - 接收策略参数：symbol, market, strategy_type, params
  - 运行回测并返回结果
  - 支持参数验证

  **Must NOT do**:
  - 不要允许危险参数

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: API端点实现简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 8, 9)
  - **Blocks**: Task 22
  - **Blocked By**: Task 8

  **References**:
  - `app.py:171-197` - API端点模式

  **Acceptance Criteria**:
  - [ ] API端点可访问
  - [ ] 参数验证正确
  - [ ] 返回回测结果JSON

  **QA Scenarios**:
  ```
  Scenario: 回测API调用
    Tool: Bash (curl)
    Steps:
      1. POST /api/backtest -d '{"symbol":"AAPL","strategy":"ma_cross"}'
      2. 验证返回回测结果
      3. 验证status code 200
    Expected Result: API返回正确回测结果
    Evidence: .sisyphus/evidence/task-10-api.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add backtest endpoint`

---

### Wave 3: Intelligence Layer - AI增强

- [ ] 11. AI记忆模块设计和实现

  **What to do**:
  - 设计分析历史存储结构（SQLite表）
  - 实现保存分析结果函数 `save_analysis(symbol, market, analysis, timestamp)`
  - 实现查询历史分析函数 `get_recent_analysis(symbol, limit=5)`
  - 实现相似股票查询功能

  **Must NOT do**:
  - 不要存储过多历史数据（限制条数）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要设计存储结构和实现多个函数
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 12, 13, 14, 15)
  - **Blocks**: Tasks 13, 14
  - **Blocked By**: Tasks 1, 5

  **References**:
  - `app.py:88-161` - AI分析函数参考

  **Acceptance Criteria**:
  - [ ] 分析结果可以保存到数据库
  - [ ] 可以查询历史分析
  - [ ] 包含时间戳和市场信息

  **QA Scenarios**:
  ```
  Scenario: AI记忆验证
    Tool: Python脚本
    Steps:
      1. 保存AAPL分析结果
      2. 查询历史分析
      3. 验证返回结果正确
    Expected Result: 记忆功能正常工作
    Evidence: .sisyphus/evidence/task-11-memory.txt
  ```

  **Commit**: YES
  - Message: `feat(ai): add analysis memory storage`

- [ ] 12. 个性化权重调整机制

  **What to do**:
  - 设计用户偏好存储结构
  - 实现记录用户反馈函数 `record_feedback(symbol, feedback_type)`
  - 实现调整分析权重函数 `adjust_analysis_weights(user_id)`
  - 基于用户反馈调整AI分析侧重点

  **Must NOT do**:
  - 不要实现复杂的推荐算法

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 权重调整逻辑相对简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 13, 14, 15)
  - **Blocks**: Task 13
  - **Blocked By**: Task 1

  **References**:
  - `app.py:88-161` - AI分析逻辑

  **Acceptance Criteria**:
  - [ ] 用户反馈可以保存
  - [ ] 分析权重可以调整
  - [ ] 调整后的分析结果有差异

  **QA Scenarios**:
  ```
  Scenario: 个性化权重调整
    Tool: Python脚本
    Steps:
      1. 记录用户对某股票的偏好
      2. 调整分析权重
      3. 验证权重变化
    Expected Result: 权重调整生效
    Evidence: .sisyphus/evidence/task-12-personalize.txt
  ```

  **Commit**: YES
  - Message: `feat(ai): add personalization weights`

- [ ] 13. 多维度分析整合（技术+基本面+情绪）

  **What to do**:
  - 设计多维度分析框架
  - 实现技术面评分函数（基于技术指标）
  - 实现基本面评分函数（基于财务数据，简化版）
  - 实现情绪面评分函数（基于新闻标题，简化版）
  - 整合三个维度给出综合评分

  **Must NOT do**:
  - 不要过度依赖基本面数据（数据源有限）
  - 不要实现复杂的NLP情感分析

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 多维度整合需要设计评分系统和权重
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 12, 14, 15)
  - **Blocks**: Task 16
  - **Blocked By**: Tasks 6, 11, 12

  **References**:
  - `app.py:88-161` - AI分析函数

  **Acceptance Criteria**:
  - [ ] 技术面评分可计算
  - [ ] 基本面评分简化实现
  - [ ] 综合评分合理
  - [ ] AI分析包含多维度

  **QA Scenarios**:
  ```
  Scenario: 多维度分析验证
    Tool: Python脚本
    Steps:
      1. 对AAPL进行多维度分析
      2. 验证三个维度评分
      3. 验证综合评分
    Expected Result: 多维度分析结果合理
    Evidence: .sisyphus/evidence/task-13-multi-dim.txt
  ```

  **Commit**: YES
  - Message: `feat(ai): add multi-dimensional analysis`

- [ ] 14. 自然语言查询接口

  **What to do**:
  - 添加Flask路由 `/api/chat`
  - 使用ZhipuAI对话能力理解用户意图
  - 提取股票代码、市场、查询类型
  - 调用相应功能并返回结果

  **Must NOT do**:
  - 不要实现复杂的意图识别系统

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: NLP处理和意图提取需要工作量
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 12, 13, 15)
  - **Blocks**: None
  - **Blocked By**: Tasks 5, 11

  **References**:
  - `app.py:141-161` - ZhipuAI调用模式

  **Acceptance Criteria**:
  - [ ] 可以理解"分析AAPL"等简单查询
  - [ ] 可以识别A股/港股代码
  - [ ] 返回正确的分析结果

  **QA Scenarios**:
  ```
  Scenario: 自然语言查询
    Tool: Bash (curl)
    Steps:
      1. POST /api/chat -d '{"message":"分析一下苹果股票"}'
      2. 验证识别AAPL
      3. 验证返回分析结果
    Expected Result: AI正确理解意图
    Evidence: .sisyphus/evidence/task-14-nlp.txt
  ```

  **Commit**: YES
  - Message: `feat(ai): add natural language query`

- [ ] 15. Prompt工程优化

  **What to do**:
  - 优化AI分析prompt模板
  - 添加结构化输出要求
  - 包含历史分析记忆
  - 添加个性化建议

  **Must NOT do**:
  - 不要让prompt过于冗长

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Prompt优化相对简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 12, 13, 14)
  - **Blocks**: Task 16
  - **Blocked By**: Task 6

  **References**:
  - `app.py:115-139` - 现有prompt模板

  **Acceptance Criteria**:
  - [ ] Prompt结构清晰
  - [ ] 输出更结构化
  - [ ] 包含历史记忆
  - [ ] AI响应质量提升

  **QA Scenarios**:
  ```
  Scenario: Prompt效果验证
    Tool: Python脚本
    Steps:
      1. 使用优化后prompt分析AAPL
      2. 验证输出结构
      3. 验证包含历史记忆
    Expected Result: 输出质量提升
    Evidence: .sisyphus/evidence/task-15-prompt.txt
  ```

  **Commit**: YES
  - Message: `feat(ai): optimize analysis prompt`

---

### Wave 4: Decision Layer - 交易信号

- [ ] 16. 交易信号生成算法

  **What to do**:
  - 实现交易信号生成函数 `generate_trading_signals(df, indicators)`
  - 基于技术指标组合生成买入/卖出/持有信号
  - 支持多种信号策略（趋势跟踪、均值回归等）
  - 返回信号强度和置信度

  **Must NOT do**:
  - 不要生成过于频繁的交易信号
  - 不要使用未来数据

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 算法设计和优化需要深入思考
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 17, 18, 19)
  - **Blocks**: Tasks F1-F4
  - **Blocked By**: Tasks 6, 8, 13, 15

  **References**:
  - `app.py:89-107` - 指标计算参考

  **Acceptance Criteria**:
  - [ ] 可以生成买入/卖出/持有信号
  - [ ] 信号包含强度和置信度
  - [ ] 支持多种策略

  **QA Scenarios**:
  ```
  Scenario: 交易信号生成
    Tool: Python脚本
    Steps:
      1. 获取AAPL数据和指标
      2. 生成交易信号
      3. 验证信号合理性
    Expected Result: 信号生成正确
    Evidence: .sisyphus/evidence/task-16-signal.txt
  ```

  **Commit**: YES
  - Message: `feat(signal): add trading signal generation`

- [ ] 17. 仓位建议计算

  **What to do**:
  - 实现仓位建议函数 `calculate_position_size(capital, risk_level, signal_strength)`
  - 基于风险等级和信号强度计算建议仓位
  - 支持保守/稳健/激进三种风险偏好
  - 返回具体金额和比例

  **Must NOT do**:
  - 不要给出过于激进的仓位建议

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 仓位计算逻辑相对简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 16, 18, 19)
  - **Blocks**: Tasks F1-F4
  - **Blocked By**: Task 16

  **References**:
  - 风险管理最佳实践

  **Acceptance Criteria**:
  - [ ] 可以计算仓位建议
  - [ ] 支持三种风险偏好
  - [ ] 建议合理

  **QA Scenarios**:
  ```
  Scenario: 仓位建议计算
    Tool: Python脚本
    Steps:
      1. 输入资金100000，稳健风险
      2. 计算仓位建议
      3. 验证建议比例合理
    Expected Result: 仓位建议合理
    Evidence: .sisyphus/evidence/task-17-position.txt
  ```

  **Commit**: YES
  - Message: `feat(signal): add position sizing`

- [ ] 18. 止损止盈点计算

  **What to do**:
  - 实现止损止盈计算函数 `calculate_stop_loss_take_profit(entry_price, volatility, risk_level)`
  - 基于ATR或波动率计算止损点
  - 根据风险偏好设置止盈点
  - 返回具体价格点位

  **Must NOT do**:
  - 不要设置过于宽松的止损点

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 止损止盈计算相对简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 16, 17, 19)
  - **Blocks**: Tasks F1-F4
  - **Blocked By**: Task 16

  **References**:
  - 风险管理最佳实践

  **Acceptance Criteria**:
  - [ ] 可以计算止损点
  - [ ] 可以计算止盈点
  - [ ] 点位合理

  **QA Scenarios**:
  ```
  Scenario: 止损止盈计算
    Tool: Python脚本
    Steps:
      1. 输入入场价和波动率
      2. 计算止损止盈点
      3. 验证点位合理性
    Expected Result: 止损止盈合理
    Evidence: .sisyphus/evidence/task-18-stop.txt
  ```

  **Commit**: YES
  - Message: `feat(signal): add stop-loss and take-profit`

- [ ] 19. 风险评估模块

  **What to do**:
  - 实现风险评估函数 `assess_risk(df, position_size)`
  - 计算最大潜在损失
  - 计算风险回报比
  - 给出风险等级和警告

  **Must NOT do**:
  - 不要给出过于乐观的风险评估

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 风险评估需要综合多个因素
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 16, 17, 18)
  - **Blocks**: Tasks F1-F4
  - **Blocked By**: Task 16

  **References**:
  - VaR和风险指标计算

  **Acceptance Criteria**:
  - [ ] 可以评估风险等级
  - [ ] 计算最大潜在损失
  - [ ] 给出风险警告

  **QA Scenarios**:
  ```
  Scenario: 风险评估
    Tool: Python脚本
    Steps:
      1. 输入数据和仓位
      2. 进行风险评估
      3. 验证风险等级
    Expected Result: 风险评估合理
    Evidence: .sisyphus/evidence/task-19-risk.txt
  ```

  **Commit**: YES
  - Message: `feat(signal): add risk assessment`

---

### Wave 5: Presentation Layer - 可视化

## Final Verification Wave

### F1. 多市场数据获取验证
**What to do**: 验证可以成功获取美股、A股、港股数据，数据格式统一，缓存机制正常工作。
**QA Scenarios**:
```
Scenario: 获取美股数据
  Tool: Python脚本
  Steps:
    1. 调用数据获取接口，symbol="AAPL", market="US"
    2. 验证返回数据包含Open/High/Low/Close/Volume
    3. 验证数据行数 >= 30天
  Expected Result: 成功获取美股数据，格式正确
  Evidence: .sisyphus/evidence/f1-us-data.txt

Scenario: 获取A股数据
  Tool: Python脚本
  Steps:
    1. 调用数据获取接口，symbol="000001.SZ", market="CN"
    2. 验证返回数据包含所需字段
    3. 验证数据来源为Akshare
  Expected Result: 成功获取A股数据，Akshare集成正常
  Evidence: .sisyphus/evidence/f1-cn-data.txt

Scenario: 数据缓存验证
  Tool: Python脚本
  Steps:
    1. 首次获取数据，记录时间
    2. 第二次获取相同数据，验证从缓存读取
    3. 验证缓存时间戳正确
  Expected Result: 第二次获取速度明显更快，缓存生效
  Evidence: .sisyphus/evidence/f1-cache.txt
```

### F2. 回测功能完整性验证
**What to do**: 验证回测框架可以运行简单策略，生成正确的回测报告和可视化图表。
**QA Scenarios**:
```
Scenario: MA交叉策略回测
  Tool: Python脚本
  Steps:
    1. 定义MA5上穿MA20买入策略
    2. 运行回测，初始资金100000
    3. 验证回测报告包含收益率、最大回撤、夏普比率
    4. 验证生成回测图表
  Expected Result: 回测成功，报告完整，图表正常显示
  Evidence: .sisyphus/evidence/f2-backtest.txt

Scenario: 回测参数配置
  Tool: Bash (curl)
  Steps:
    1. POST /api/backtest，传入策略参数
    2. 验证返回JSON包含回测结果
    3. 验证参数配置生效
  Expected Result: API返回正确的回测结果
  Evidence: .sisyphus/evidence/f2-api.txt
```

### F3. 自然语言交互验证
**What to do**: 验证可以通过自然语言查询股票，AI能理解用户意图并返回正确结果。
**QA Scenarios**:
```
Scenario: 自然语言查询美股
  Tool: Bash (curl)
  Steps:
    1. POST /api/chat，message="分析一下苹果股票最近的走势"
    2. 验证AI识别股票代码为AAPL
    3. 验证返回分析结果
  Expected Result: AI正确理解意图，返回AAPL分析
  Evidence: .sisyphus/evidence/f3-nlp-us.txt

Scenario: 自然语言查询A股
  Tool: Bash (curl)
  Steps:
    1. POST /api/chat，message="贵州茅台最近怎么样"
    2. 验证AI识别A股代码
    3. 验证调用A股数据源
  Expected Result: AI识别茅台股票代码，返回A股数据
  Evidence: .sisyphus/evidence/f3-nlp-cn.txt
```

### F4. 整体功能集成测试
**What to do**: 验证所有模块集成后系统稳定，各功能协同工作正常。
**QA Scenarios**:
```
Scenario: 完整分析流程
  Tool: Python脚本
  Steps:
    1. 自然语言查询"AAPL最近走势"
    2. 系统获取数据 → 计算指标 → 运行回测 → AI分析 → 生成信号
    3. 验证每一步输出正确
    4. 验证可视化图表生成
  Expected Result: 完整流程顺畅，各模块协同正常
  Evidence: .sisyphus/evidence/f4-integration.txt

Scenario: AI记忆验证
  Tool: Python脚本
  Steps:
    1. 首次查询"AAPL分析"
    2. 第二次查询"上次的分析结果"
    3. 验证AI能回忆上次分析
  Expected Result: AI成功回忆历史分析
  Evidence: .sisyphus/evidence/f4-memory.txt
```

---

## Commit Strategy

- **Wave 1完成**: `feat(data): add multi-market data fetching with cache`
- **Wave 2完成**: `feat(analysis): add technical indicators and backtesting`
- **Wave 3完成**: `feat(ai): add memory and personalization`
- **Wave 4完成**: `feat(signal): add trading signal generation`
- **Wave 5完成**: `feat(ui): add Plotly visualization dashboard`
- **Final**: `feat: complete smart stock analysis platform iteration`

---

## Success Criteria

### Verification Commands
```bash
# 启动应用
python app.py

# 测试多市场数据获取
python test/test_multi_market.py

# 测试回测功能
python test/test_backtest.py

# 测试自然语言交互
python test/test_nlp.py

# 手动验证可视化
# 访问 http://localhost:5000/dashboard
```

### Final Checklist
- [ ] 可以获取美股、A股、港股数据
- [ ] 可以计算20+技术指标
- [ ] 可以运行策略回测
- [ ] 自然语言查询正常工作
- [ ] AI记忆功能正常
- [ ] 可视化图表正确显示
- [ ] 自动生成交易信号
- [ ] 所有手动测试通过

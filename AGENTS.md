# AGENTS.md - AI Trade Project Guide

## Project Overview

This is a Python Flask web application that provides AI-powered stock market analysis. It uses the ZhipuAI (智谱AI) API to analyze stock data retrieved from Yahoo Finance.

**Core Technologies:**
- Python 3.13+
- Flask 2.3.3 - Web framework
- yfinance 0.2.65 - Stock data retrieval
- zhipuai 2.1.5 - AI analysis (Chinese LLM provider)
- pandas/numpy - Data processing

**Deployment:** Vercel (serverless)

## Project Structure

```
ai_trade/
├── app.py                  # Main Flask application (single-file architecture)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (ZHIPUAI_API_KEY)
├── templates/              # HTML templates
│   ├── index.html         # Main page
│   └── about.html         # About page
├── test/                   # Manual test scripts (not pytest)
│   ├── test_yfinance.py
│   └── test_optimized_yfinance.py
└── .venv/                  # Virtual environment (gitignored)
```

## Build & Run Commands

### Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the project root:
```
ZHIPUAI_API_KEY=your_api_key_here
```

### Run Application
```bash
# Development mode (with debug enabled)
python app.py

# The app runs on http://localhost:5000 by default
```

### Testing
```bash
# Run manual test scripts
python test/test_yfinance.py
python test/test_optimized_yfinance.py

# Note: These are manual test scripts, not pytest suites
# They print output to console for verification
```

### Deployment
```bash
# Vercel deployment (if Vercel CLI is installed)
vercel --prod
```

## Code Style Guidelines

### Python Style

**Imports Organization:**
```python
# 1. Standard library imports
import os
import sys
import textwrap
import traceback
from datetime import datetime, timedelta

# 2. Third-party imports
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# 3. Local imports (if any)
```

**Naming Conventions:**
- Functions: `snake_case` (e.g., `fetch_stock_data`, `analyze_stock_with_ai`)
- Variables: `snake_case` (e.g., `api_key`, `max_retries`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `INITIAL_DELAY`)
- Flask routes: Simple function names matching route purpose

**String Formatting:**
- Use f-strings for all string interpolation
- Use `textwrap.dedent()` for multi-line prompts/templates
```python
prompt = textwrap.dedent(f"""
    作为一名专业股票分析师，请分析以下 {symbol} 股票数据...
    """)
```

**Comments:**
- This project uses Chinese comments (中文注释)
- Continue this pattern for consistency
- Use emoji indicators for status messages: ✅ (success), ⚠️ (warning), 🚨 (error)

**Error Handling:**
```python
# Use specific exception types
try:
    # operation
except SpecificException as e:
    error_msg = f"🚨 Error description: {str(e)}"
    print(f"{error_msg}: {traceback.format_exc()}")
    return error_msg
```

**Logging:**
- Use print statements with emoji indicators
- Include context in error messages
- Log API call status and retry attempts

### Flask Routes

**Route Structure:**
```python
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/endpoint', methods=['POST'])
def api_endpoint():
    try:
        data = request.json
        # Process data
        return jsonify({"result": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

**Response Format:**
- Success: `jsonify({"key": value})`
- Error: `jsonify({"error": "message"}), status_code`

### Data Processing

**Pandas Operations:**
```python
# Check for required columns
if 'Adj Close' not in data.columns:
    if 'Close' in data.columns:
        data['Adj Close'] = data['Close']

# Create derived columns
data['Daily Return'] = data['Adj Close'].pct_change()
data['MA5'] = data['Adj Close'].rolling(window=5).mean()
```

**API Retry Logic:**
- Use exponential backoff for rate-limited APIs
- Maximum 5 retries with increasing delays
- Provide fallback behavior (e.g., simulated data)

### Type Hints

While the current codebase doesn't use type hints, adding them for new code is encouraged:
```python
def fetch_stock_data(symbol: str, days: int = 30) -> pd.DataFrame:
    ...

def analyze_stock_with_ai(stock_data: pd.DataFrame, symbol: str) -> str:
    ...
```

## Error Handling Patterns

### API Rate Limiting
```python
max_retries = 5
initial_delay = 2
backoff_factor = 2

for attempt in range(max_retries):
    try:
        if attempt > 0:
            delay = initial_delay * (backoff_factor ** attempt) + random.uniform(0, attempt)
            time.sleep(delay)
        # API call
    except YFRateLimitError:
        print(f"⚠️ Rate limit error ({attempt + 1}/{max_retries})")
```

### Empty Data Handling
```python
if df.empty:
    return jsonify({"error": "未获取到数据"}), 404
```

### Graceful Degradation
- When real data unavailable, use simulated data
- Log fallback behavior clearly
- Maintain functionality even when APIs fail

## Testing Guidelines

### Current Testing Approach
- Manual test scripts in `test/` directory
- Print-based verification
- No automated test framework

### Running Tests
```bash
# Test basic yfinance connectivity
python test/test_yfinance.py

# Test optimized retry logic
python test/test_optimized_yfinance.py
```

### Adding New Tests
For new functionality, create manual test scripts following the existing pattern:
```python
# test/test_new_feature.py
import sys
sys.path.append('..')
from app import function_to_test

print("Testing new feature...")
try:
    result = function_to_test()
    print(f"✅ Success: {result}")
except Exception as e:
    print(f"❌ Error: {e}")
```

## Environment Variables

Required environment variables in `.env`:
- `ZHIPUAI_API_KEY` - API key for ZhipuAI service

## Important Notes

1. **Single-File Architecture:** The main application logic is in `app.py`. For larger features, consider modularizing into separate modules.

2. **API Dependencies:** The app depends on external APIs (Yahoo Finance, ZhipuAI). Always implement retry logic and fallbacks.

3. **Chinese Language:** Comments and user-facing messages use Chinese. Maintain this for consistency.

4. **No Linter Config:** The project doesn't have explicit linter configuration. Follow PEP 8 and match existing code style.

5. **Vercel Deployment:** The app is deployed on Vercel. Ensure compatibility with serverless environment.

6. **Virtual Environment:** Always activate `.venv` before running or testing.

## Common Tasks

### Add a New API Endpoint
1. Add route decorator in `app.py`
2. Implement error handling with try/except
3. Return JSON responses
4. Test manually with curl or Postman

### Add New Stock Analysis Feature
1. Implement analysis function following existing pattern
2. Add technical indicators using pandas
3. Update prompt template for AI analysis
4. Handle edge cases (missing data, API errors)

### Update Dependencies
```bash
pip install new_package
pip freeze > requirements.txt
```

## AI Agent Working Guidelines

When working in this codebase:

1. **Maintain Consistency:** Follow existing patterns for error handling, logging, and API interactions.

2. **Chinese Comments:** Use Chinese for comments and user-facing messages to match project style.

3. **Error Recovery:** Always implement retry logic for external API calls with exponential backoff.

4. **No Type Suppression:** Never use `# type: ignore` or `except: pass`. Handle errors explicitly.

5. **Test Changes:** Run manual test scripts after making changes to verify functionality.

6. **Virtual Environment:** Ensure virtual environment is activated before running any Python commands.

7. **Environment Variables:** Never commit `.env` file. Use `.env.example` for documentation if needed.

8. **Minimal Changes:** When fixing bugs, make minimal changes to address the specific issue without refactoring surrounding code.

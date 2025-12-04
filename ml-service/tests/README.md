# ML Engine Test Suite

Unit and integration tests for the **Nutri ML Engine**.

## Overview

This test suite validates the ML Engine across all phases:

```
📊 Phase 1: Feature Engineering & Correlation Analysis
    ↓
🧠 Phase 2: PyTorch LSTM Training & Predictions
    ↓
🔍 Phase 3: Model Interpretability & Explainability
```

## Test Files

| File | Purpose |
|------|---------|
| `fixtures.py` | `TestDataGenerator` - Creates realistic test data |
| `conftest.py` | Pytest configuration and shared fixtures |
| `test_auth.py` | Authentication tests |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/test_auth.py -v

# Run with verbose output
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -v -x
```

## Test Data Generation

Located in `fixtures.py`, the `TestDataGenerator` class generates realistic test data:

```python
from tests.fixtures import TestDataGenerator

generator = TestDataGenerator(seed=42)
dataset = generator.generate_complete_dataset()
```

## Debugging

### View SQL Queries

Edit `conftest.py` and set `echo=True`:

```python
engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    echo=True,  # Shows all SQL queries
)
```

### Common Issues

**Issue**: `ImportError: No module named 'app'`
```bash
# Solution: Run from ml-service directory
cd ml-service
pytest tests/ -v
```

**Issue**: Tests hang or timeout
```bash
# Solution: Increase timeout in test
async with AsyncClient(app=app, timeout=600.0) as client:
    ...
```

## Adding New Tests

```python
# tests/test_new_feature.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_new_feature(db):
    """Test description."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/endpoint", json={...})
    
    assert response.status_code == 200
```

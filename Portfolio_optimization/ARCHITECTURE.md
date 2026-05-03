# Portfolio Optimization Platform Architecture

## Overview

This platform implements a **Clean Architecture** with **Domain-Driven Design** principles for quantitative portfolio optimization.

## Directory Structure

```
src/
├── domain/                    # Core business logic (independent of frameworks)
│   ├── models.py             # Domain models (OptimizationResult, PortfolioMetrics)
│   ├── value_objects.py      # Immutable value objects (Ticker, Weight, etc.)
│   └── interfaces.py         # Protocol definitions (PortfolioOptimizer, etc.)
│
├── application/               # Application-specific business rules
│   ├── dto.py                # Data Transfer Objects
│   ├── config.py             # Experiment configuration management
│   ├── services/
│   │   └── optimizers.py     # Optimization strategy implementations
│   └── use_cases/
│       └── optimize_portfolio.py  # Main optimization use case
│
├── infrastructure/            # External dependencies & adapters
│   └── moex/
│       └── data_provider.py  # MOEX API implementation
│
├── analytics/                 # Advanced analytics & research
│   ├── risk/
│   │   └── cvar.py           # CVaR, Ledoit-Wolf, diagnostics
│   └── backtesting/
│       └── walk_forward.py   # Walk-forward backtesting engine
│
├── tests/                     # Unit and integration tests
│   └── test_optimizers.py    # Optimizer unit tests
│
└── [legacy]                   # Original modules (for backward compatibility)
    ├── models.py             # Legacy PortfolioMath class
    ├── data_loader.py        # Legacy MoexDataLoader
    ├── visuals.py            # Plotly visualizations
    └── expert_engine.py      # Expert analysis reports
```

## Key Design Patterns

### 1. Strategy Pattern
Each optimization algorithm implements the `BaseOptimizer` interface:
- `MarkowitzOptimizer` - Max Sharpe Ratio
- `RiskParityOptimizer` - Equal Risk Contribution
- `MinimumVarianceOptimizer` - Global Minimum Variance
- `MaximumDiversificationOptimizer` - Max Diversification Ratio
- `CVaROptimizer` - Minimum Tail Risk
- `HRPOptimizer` - Hierarchical Risk Parity

### 2. Factory Pattern
```python
from src.application.services.optimizers import get_optimizer
optimizer = get_optimizer('max_sharpe', risk_free_rate=0.16)
weights = optimizer.optimize(returns)
```

### 3. Use Case Pattern
```python
from src.application.use_cases import PortfolioOptimizationUseCase
use_case = PortfolioOptimizationUseCase(returns, risk_free_rate=0.16)
use_case.set_strategy('hrp')
response = use_case.execute()
```

### 4. Repository Pattern (Data Access)
```python
from src.infrastructure.moex import MoexDataProvider
provider = MoexDataProvider()
returns = provider.fetch_returns(tickers, start, end)
```

## Configuration Management

Experiments are reproducible via YAML configuration:

```yaml
# experiment_configs/example.yaml
market: moex
tickers: [SBER, LKOH, GAZP]
strategy: risk_parity
risk_free_rate: 0.16
train_window: 252
rebalance_period: 21
```

Load with:
```python
from src.application.config import ExperimentConfig
config = ExperimentConfig.from_yaml('experiment_configs/example.yaml')
```

## Testing

Run all tests:
```bash
pytest src/tests/ -v
```

## Quality Tools

- **ruff**: Fast Python linter
- **mypy**: Static type checking
- **pytest**: Unit testing framework

Configure in `pyproject.toml`.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
- Tests with coverage
- Linting with ruff
- Type checking with mypy

## Migration Guide

### Old Code (Legacy)
```python
from src.models import PortfolioMath
math = PortfolioMath(returns, risk_free_rate=0.16)
weights = math.optimize_sharpe()
metrics = math.calc_metrics(weights)
```

### New Code (Recommended)
```python
from src.application.services.optimizers import MarkowitzOptimizer
from src.analytics.risk import RiskAnalytics

optimizer = MarkowitzOptimizer(risk_free_rate=0.16)
weights = optimizer.optimize(returns)

# Calculate metrics
port_return = np.sum(returns.mean() * 252 * weights)
port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
```

Both approaches work; new architecture is recommended for:
- Better testability
- Clearer separation of concerns
- Easier extension with new strategies
- Reproducible experiments

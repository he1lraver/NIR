"""
Value objects for the portfolio optimization domain.

Value objects are immutable objects that represent concepts by their attributes.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class Ticker:
    """Represents a stock ticker symbol."""
    symbol: str
    
    def __post_init__(self):
        object.__setattr__(self, 'symbol', self.symbol.upper().strip())


@dataclass(frozen=True)
class DateRange:
    """Represents a date range for data fetching."""
    start: str  # YYYY-MM-DD format
    end: str    # YYYY-MM-DD format


@dataclass(frozen=True)
class RiskFreeRate:
    """Represents annual risk-free rate as a decimal."""
    value: float  # e.g., 0.16 for 16%
    
    @property
    def daily(self) -> float:
        """Convert annual rate to daily rate (252 trading days)."""
        return self.value / 252


@dataclass(frozen=True)
class Weight:
    """Represents a portfolio weight for a single asset."""
    ticker: Ticker
    value: float  # 0.0 to 1.0
    
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Weight must be between 0 and 1, got {self.value}")


@dataclass
class PortfolioWeights:
    """Collection of portfolio weights that sum to 1."""
    weights: List[Weight]
    
    @property
    def tickers(self) -> List[str]:
        return [w.ticker.symbol for w in self.weights]
    
    @property
    def values(self) -> np.ndarray:
        return np.array([w.value for w in self.weights])
    
    @property
    def sum(self) -> float:
        return sum(w.value for w in self.weights)
    
    def to_dict(self) -> dict:
        return {w.ticker.symbol: w.value for w in self.weights}
    
    def validate(self, tolerance: float = 1e-6) -> bool:
        """Validate that weights sum to 1 within tolerance."""
        return abs(self.sum - 1.0) < tolerance


@dataclass(frozen=True)
class AssetReturns:
    """Time series of asset returns."""
    data: any  # pandas DataFrame with dates as index, tickers as columns
    
    @property
    def tickers(self) -> List[str]:
        return list(self.data.columns)
    
    @property
    def n_assets(self) -> int:
        return len(self.tickers)
    
    @property
    def n_observations(self) -> int:
        return len(self.data)

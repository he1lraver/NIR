"""
Domain models for portfolio optimization results and metrics.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    """
    Immutable result of a portfolio optimization run.
    
    This value object encapsulates all key metrics from an optimization,
    ensuring reproducibility and type safety.
    """
    weights: np.ndarray
    tickers: List[str]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    
    @property
    def weights_dict(self) -> Dict[str, float]:
        """Return weights as a dictionary mapping ticker to weight."""
        return dict(zip(self.tickers, self.weights))
    
    @property
    def n_assets(self) -> int:
        """Number of assets in the portfolio."""
        return len(self.tickers)
    
    @property
    def active_weights(self) -> Dict[str, float]:
        """Return only non-zero weights (> 0.1%)."""
        return {k: v for k, v in self.weights_dict.items() if v > 0.001}
    
    def validate(self, tolerance: float = 1e-6) -> bool:
        """Validate that weights sum to 1 within tolerance."""
        return abs(np.sum(self.weights) - 1.0) < tolerance
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'weights': self.weights_dict,
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'max_drawdown': self.max_drawdown,
            'n_assets': self.n_assets
        }


@dataclass
class PortfolioMetrics:
    """
    Mutable container for detailed portfolio metrics.
    
    Includes both summary statistics and time-series data for analysis.
    """
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    daily_returns: np.ndarray
    drawdowns: np.ndarray
    dates: Optional[np.ndarray] = None
    
    def to_summary_dict(self) -> dict:
        """Return summary metrics as dictionary."""
        return {
            'return': self.expected_return,
            'volatility': self.volatility,
            'sharpe': self.sharpe_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'max_drawdown': self.max_drawdown
        }


@dataclass
class RiskContribution:
    """
    Risk contribution breakdown for each asset.
    
    Used primarily for Risk Parity strategies.
    """
    ticker: str
    weight: float
    marginal_risk: float
    risk_contribution: float
    target_contribution: float
    
    @property
    def deviation_from_target(self) -> float:
        """Absolute deviation from target risk contribution."""
        return abs(self.risk_contribution - self.target_contribution)

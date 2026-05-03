"""
Domain interfaces (protocols) for portfolio optimization.

These define contracts that concrete implementations must follow.
"""
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import numpy as np
import pandas as pd


@runtime_checkable
class PortfolioOptimizer(Protocol):
    """Protocol for portfolio optimization strategies."""
    
    @abstractmethod
    def optimize(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Optimize portfolio weights given historical returns.
        
        Args:
            returns: DataFrame with daily returns (dates x assets)
            
        Returns:
            Array of optimal weights summing to 1.0
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the optimization strategy."""
        pass


@runtime_checkable
class RiskCalculator(Protocol):
    """Protocol for risk metric calculations."""
    
    @abstractmethod
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        pass
    
    @abstractmethod
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        pass
    
    @abstractmethod
    def calculate_volatility(self, returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate volatility (standard deviation)."""
        pass


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for market data providers."""
    
    @abstractmethod
    def fetch_prices(self, tickers: list, start: str, end: str) -> pd.DataFrame:
        """Fetch historical prices for given tickers."""
        pass
    
    @abstractmethod
    def fetch_returns(self, tickers: list, start: str, end: str) -> pd.DataFrame:
        """Fetch historical returns for given tickers."""
        pass

"""
DTOs (Data Transfer Objects) for the application layer.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import date


@dataclass
class OptimizationRequest:
    """Request DTO for portfolio optimization."""
    tickers: List[str]
    start_date: date
    end_date: date
    risk_free_rate: float  # Annual rate as decimal (e.g., 0.16 for 16%)
    strategy: str  # e.g., 'max_sharpe', 'risk_parity', 'hrp', etc.
    
    def validate(self) -> tuple[bool, str]:
        """Validate the request parameters."""
        if len(self.tickers) < 2:
            return False, "Minimum 2 tickers required"
        if self.start_date >= self.end_date:
            return False, "Start date must be before end date"
        if not 0 <= self.risk_free_rate <= 1:
            return False, "Risk-free rate must be between 0 and 1"
        valid_strategies = ['max_sharpe', 'risk_parity', 'min_vol', 
                           'max_div', 'min_cvar', 'hrp']
        if self.strategy not in valid_strategies:
            return False, f"Strategy must be one of {valid_strategies}"
        return True, "OK"


@dataclass
class OptimizationResponse:
    """Response DTO for portfolio optimization results."""
    success: bool
    weights: Optional[Dict[str, float]] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    @classmethod
    def success_response(cls, weights: Dict[str, float], 
                         metrics: Dict[str, float],
                         warnings: Optional[List[str]] = None) -> 'OptimizationResponse':
        return cls(
            success=True,
            weights=weights,
            metrics=metrics,
            warnings=warnings or []
        )
    
    @classmethod
    def failure_response(cls, error: str) -> 'OptimizationResponse':
        return cls(
            success=False,
            error_message=error
        )

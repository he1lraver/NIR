"""
Use cases for portfolio optimization.

Use cases orchestrate domain objects and services to accomplish specific tasks.
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from datetime import date

from ..dto import OptimizationRequest, OptimizationResponse
from .optimizers import get_optimizer, BaseOptimizer


class PortfolioOptimizationUseCase:
    """
    Main use case for portfolio optimization.
    
    Orchestrates data loading, optimization, and metric calculation.
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.05):
        """
        Initialize the use case.
        
        Args:
            returns: DataFrame with daily returns (dates x assets)
            risk_free_rate: Annual risk-free rate as decimal
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self._optimizer: Optional[BaseOptimizer] = None
    
    def set_strategy(self, strategy: str) -> None:
        """Set the optimization strategy."""
        self._optimizer = get_optimizer(strategy, self.risk_free_rate)
    
    def execute(self) -> OptimizationResponse:
        """
        Execute the optimization use case.
        
        Returns:
            OptimizationResponse with weights and metrics
        """
        if self._optimizer is None:
            return OptimizationResponse.failure_response(
                "No optimization strategy selected"
            )
        
        try:
            # Run optimization
            weights = self._optimizer.optimize(self.returns)
            
            # Validate weights
            if not self._validate_weights(weights):
                return OptimizationResponse.failure_response(
                    "Optimization failed: invalid weights"
                )
            
            # Calculate metrics
            metrics = self._calculate_metrics(weights)
            
            # Build response
            weights_dict = dict(zip(self.returns.columns, weights))
            
            warnings = []
            if self._check_numerical_stability(warnings):
                pass  # Warnings already populated
            
            return OptimizationResponse.success_response(
                weights=weights_dict,
                metrics=metrics,
                warnings=warnings
            )
            
        except Exception as e:
            return OptimizationResponse.failure_response(str(e))
    
    def _validate_weights(self, weights: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Validate that weights sum to 1 and are non-negative."""
        return (abs(np.sum(weights) - 1.0) < tolerance and 
                np.all(weights >= -tolerance))
    
    def _calculate_metrics(self, weights: np.ndarray) -> dict:
        """Calculate portfolio metrics."""
        # Annualized metrics
        mean_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        port_return = np.sum(mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        rfr_annual = self.risk_free_rate
        sharpe = (port_return - rfr_annual) / port_vol if port_vol > 0 else 0
        
        # Daily returns for tail risk calculations
        daily_returns = self.returns.dot(weights)
        
        # VaR and CVaR (95%)
        var_95 = -np.percentile(daily_returns, 5)
        tail_losses = daily_returns[daily_returns <= -var_95]
        cvar_95 = -tail_losses.mean() if len(tail_losses) > 0 else var_95
        
        # Annualize tail risks
        var_95_annual = var_95 * np.sqrt(252)
        cvar_95_annual = cvar_95 * np.sqrt(252)
        
        # Maximum drawdown
        cum_returns = (1 + daily_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        return {
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'var_95': var_95_annual,
            'cvar_95': cvar_95_annual,
            'max_drawdown': max_dd
        }
    
    def _check_numerical_stability(self, warnings: list) -> bool:
        """Check for numerical stability issues."""
        has_issues = False
        
        # Check covariance matrix condition number
        cov_matrix = self.returns.cov().values
        try:
            cond_num = np.linalg.cond(cov_matrix)
            if cond_num > 1e10:
                warnings.append(
                    f"High condition number ({cond_num:.2e}): "
                    "covariance matrix may be ill-conditioned"
                )
                has_issues = True
        except Exception:
            pass
        
        # Check for negative eigenvalues
        try:
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            if np.any(eigenvalues < -1e-10):
                warnings.append(
                    "Covariance matrix has negative eigenvalues: "
                    "not positive semi-definite"
                )
                has_issues = True
        except Exception:
            pass
        
        return has_issues


def run_optimization(request: OptimizationRequest, 
                     returns: pd.DataFrame) -> OptimizationResponse:
    """
    Convenience function to run optimization from request DTO.
    
    Args:
        request: OptimizationRequest DTO
        returns: Pre-fetched returns DataFrame
        
    Returns:
        OptimizationResponse
    """
    # Validate request
    is_valid, message = request.validate()
    if not is_valid:
        return OptimizationResponse.failure_response(message)
    
    # Create and execute use case
    use_case = PortfolioOptimizationUseCase(returns, request.risk_free_rate)
    use_case.set_strategy(request.strategy)
    
    return use_case.execute()

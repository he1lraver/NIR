"""
Portfolio optimization strategies implementing the PortfolioOptimizer protocol.

Each strategy is a separate class following the Single Responsibility Principle.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List


class BaseOptimizer(ABC):
    """Abstract base class for portfolio optimizers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def optimize(self, returns: pd.DataFrame) -> np.ndarray:
        pass
    
    def _validate_weights(self, weights: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Validate that weights sum to 1 and are non-negative."""
        return (abs(np.sum(weights) - 1.0) < tolerance and 
                np.all(weights >= -tolerance))


class MarkowitzOptimizer(BaseOptimizer):
    """
    Modern Portfolio Theory optimizer (Max Sharpe Ratio).
    
    Maximizes the Sharpe ratio: (E[R_p] - R_f) / σ_p
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    @property
    def name(self) -> str:
        return "Max Sharpe Ratio (Markowitz MPT)"
    
    def optimize(self, returns: pd.DataFrame) -> np.ndarray:
        n_assets = len(returns.columns)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        rfr_daily = self.risk_free_rate / 252
        
        def neg_sharpe(weights):
            port_return = np.sum(mean_returns * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_vol == 0:
                return 0
            return -(port_return - rfr_daily * 252) / port_vol
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        result = minimize(
            neg_sharpe,
            x0=n_assets * [1.0 / n_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x


class RiskParityOptimizer(BaseOptimizer):
    """
    Risk Parity optimizer (Equal Risk Contribution).
    
    Each asset contributes equally to total portfolio risk.
    """
    
    @property
    def name(self) -> str:
        return "Risk Parity (All Weather)"
    
    def optimize(self, returns: pd.DataFrame) -> np.ndarray:
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        
        def risk_budget_objective(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_vol == 0:
                return 0
            mrc = np.dot(cov_matrix, weights) / port_vol
            rc = weights * mrc
            target_rc = port_vol / n_assets
            return np.sum(np.square(rc - target_rc))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        result = minimize(
            risk_budget_objective,
            x0=n_assets * [1.0 / n_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x


class MinimumVarianceOptimizer(BaseOptimizer):
    """
    Global Minimum Variance Portfolio.
    
    Minimizes portfolio volatility: min w'Σw
    """
    
    @property
    def name(self) -> str:
        return "Minimum Variance"
    
    def optimize(self, returns: pd.DataFrame) -> np.ndarray:
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        
        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        result = minimize(
            portfolio_vol,
            x0=n_assets * [1.0 / n_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x


class MaximumDiversificationOptimizer(BaseOptimizer):
    """
    Maximum Diversification Portfolio.
    
    Maximizes diversification ratio: DR(w) = (Σw_iσ_i) / σ_p
    """
    
    @property
    def name(self) -> str:
        return "Maximum Diversification"
    
    def optimize(self, returns: pd.DataFrame) -> np.ndarray:
        n_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        vols = returns.std() * np.sqrt(252)
        
        def neg_diversification_ratio(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_vol == 0:
                return 0
            weighted_vols = np.dot(weights, vols)
            return -(weighted_vols / port_vol)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        result = minimize(
            neg_diversification_ratio,
            x0=n_assets * [1.0 / n_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x


class CVaROptimizer(BaseOptimizer):
    """
    Conditional Value at Risk optimizer.
    
    Minimizes CVaR (Expected Shortfall) at 95% confidence.
    Note: This uses empirical CVaR; for Rockafellar-Uryasev LP formulation,
    see the analytics/risk module.
    """
    
    @property
    def name(self) -> str:
        return "Minimum CVaR"
    
    def optimize(self, returns: pd.DataFrame) -> np.ndarray:
        n_assets = len(returns.columns)
        
        def cvar_objective(weights):
            port_returns = returns.dot(weights)
            # Empirical CVaR at 95%
            var_95 = np.percentile(port_returns, 5)
            tail_losses = port_returns[port_returns <= var_95]
            if len(tail_losses) == 0:
                return -var_95
            return -tail_losses.mean()
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        result = minimize(
            cvar_objective,
            x0=n_assets * [1.0 / n_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x


class HRPOptimizer(BaseOptimizer):
    """
    Hierarchical Risk Parity (Lopez de Prado, 2016).
    
    Uses hierarchical clustering to allocate weights based on
    the tree structure of asset correlations.
    
    Algorithm steps:
    1. Compute correlation matrix
    2. Convert to distance matrix: d = sqrt((1 - ρ) / 2)
    3. Perform hierarchical clustering (single linkage)
    4. Quasi-diagonalization (sort leaves)
    5. Recursive bisection to allocate weights
    """
    
    @property
    def name(self) -> str:
        return "Hierarchical Risk Parity (ML)"
    
    def optimize(self, returns: pd.DataFrame) -> np.ndarray:
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import squareform
        
        corr = returns.corr().values
        
        # Step 1-2: Convert correlation to distance
        dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
        condensed_dist = squareform(dist, checks=False)
        
        # Step 3: Hierarchical clustering
        link = sch.linkage(condensed_dist, method='single')
        sort_ix = sch.leaves_list(link)
        
        # Step 4-5: Recursive bisection
        cov_matrix = returns.cov().values * 252
        
        def recursive_bisection(cov, sorted_indices):
            n = len(sorted_indices)
            weights = pd.Series(1.0, index=sorted_indices)
            clusters = [sorted_indices]
            
            while len(clusters) > 0:
                # Split each cluster in half
                new_clusters = []
                for cluster in clusters:
                    mid = len(cluster) // 2
                    if mid > 0:
                        new_clusters.append(cluster[:mid])
                        new_clusters.append(cluster[mid:])
                
                clusters = new_clusters
                
                # Process pairs of clusters
                for i in range(0, len(clusters) - 1, 2):
                    c1 = clusters[i]
                    c2 = clusters[i + 1]
                    
                    # Calculate variance of each cluster
                    var_c1 = np.dot(weights[c1].T, 
                                   np.dot(cov[np.ix_(c1, c1)], weights[c1]))
                    var_c2 = np.dot(weights[c2].T, 
                                   np.dot(cov[np.ix_(c2, c2)], weights[c2]))
                    
                    # Allocate alpha inversely proportional to variance
                    alpha = 1 - var_c1 / (var_c1 + var_c2) if (var_c1 + var_c2) > 0 else 0.5
                    weights[c1] *= alpha
                    weights[c2] *= (1 - alpha)
            
            return weights
        
        weights = recursive_bisection(cov_matrix, sort_ix)
        
        # Reorder to original ticker order
        final_weights = np.zeros(len(returns.columns))
        for i, idx in enumerate(sort_ix):
            final_weights[idx] = weights.iloc[i]
        
        # Normalize
        return final_weights / np.sum(final_weights)


def get_optimizer(strategy: str, risk_free_rate: float = 0.05) -> BaseOptimizer:
    """Factory function to get optimizer by strategy name."""
    optimizers = {
        'max_sharpe': MarkowitzOptimizer(risk_free_rate),
        'risk_parity': RiskParityOptimizer(),
        'min_vol': MinimumVarianceOptimizer(),
        'max_div': MaximumDiversificationOptimizer(),
        'min_cvar': CVaROptimizer(),
        'hrp': HRPOptimizer()
    }
    
    if strategy not in optimizers:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Available: {list(optimizers.keys())}")
    
    return optimizers[strategy]

"""
Backtesting module for walk-forward analysis.

Implements proper train/test split with rebalancing as described in:
- Advances in Financial Machine Learning (Lopez de Prado, 2018)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Results from a walk-forward backtest."""
    returns: pd.Series
    weights_history: pd.DataFrame
    metrics: Dict[str, float]
    trades: int


class WalkForwardBacktester:
    """
    Walk-forward backtesting engine.
    
    Implements proper out-of-sample testing with periodic rebalancing.
    """
    
    def __init__(self, 
                 optimizer,
                 train_window: int = 252,
                 rebalance_period: int = 21,
                 min_train_samples: int = 126):
        """
        Initialize backtester.
        
        Args:
            optimizer: Portfolio optimizer instance
            train_window: Number of days for training window
            rebalance_period: Days between rebalancing
            min_train_samples: Minimum samples required for first training
        """
        self.optimizer = optimizer
        self.train_window = train_window
        self.rebalance_period = rebalance_period
        self.min_train_samples = min_train_samples
    
    def run(self, returns: pd.DataFrame) -> BacktestResult:
        """
        Run walk-forward backtest.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            BacktestResult with performance metrics
        """
        n_obs = len(returns)
        tickers = returns.columns.tolist()
        
        if n_obs < self.min_train_samples + self.rebalance_period:
            raise ValueError("Insufficient data for backtesting")
        
        # Storage for results
        portfolio_returns = []
        weights_history = []
        dates = []
        trades = 0
        
        current_weights = None
        
        # Walk forward through time
        for i in range(self.min_train_samples, n_obs, self.rebalance_period):
            # Training period
            train_start = max(0, i - self.train_window)
            train_returns = returns.iloc[train_start:i]
            
            # Optimize
            try:
                weights = self.optimizer.optimize(train_returns)
                current_weights = weights
                trades += 1
            except Exception:
                # Keep previous weights if optimization fails
                if current_weights is None:
                    current_weights = np.ones(len(tickers)) / len(tickers)
            
            # Apply weights to next period returns
            end_idx = min(i + self.rebalance_period, n_obs)
            period_returns = returns.iloc[i:end_idx]
            
            for date, row in period_returns.iterrows():
                port_ret = np.dot(row.values, current_weights)
                portfolio_returns.append(port_ret)
                weights_history.append(current_weights.copy())
                dates.append(date)
        
        # Build results
        returns_series = pd.Series(portfolio_returns, index=dates, name='portfolio')
        weights_df = pd.DataFrame(weights_history, index=dates, columns=tickers)
        
        # Calculate metrics
        metrics = self._calculate_metrics(returns_series)
        
        return BacktestResult(
            returns=returns_series,
            weights_history=weights_df,
            metrics=metrics,
            trades=trades
        )
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Annualized return
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Annualized volatility
        ann_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'n_observations': len(returns)
        }


class BenchmarkComparator:
    """
    Compare portfolio strategies against benchmarks.
    """
    
    @staticmethod
    def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
        """Calculate returns of equal-weight portfolio."""
        n_assets = len(returns.columns)
        weights = np.ones(n_assets) / n_assets
        return returns.dot(weights)
    
    @staticmethod
    def random_portfolios(returns: pd.DataFrame, 
                          n_portfolios: int = 1000) -> pd.DataFrame:
        """Generate random portfolio returns for comparison."""
        n_assets = len(returns.columns)
        random_returns = []
        
        for _ in range(n_portfolios):
            # Random Dirichlet weights (sum to 1)
            weights = np.random.dirichlet(np.ones(n_assets))
            port_returns = returns.dot(weights)
            random_returns.append(port_returns)
        
        return pd.DataFrame(random_returns).T
    
    @staticmethod
    def compare_strategies(returns: pd.DataFrame,
                           strategies: Dict[str, any]) -> pd.DataFrame:
        """
        Compare multiple strategies against benchmarks.
        
        Args:
            returns: Asset returns DataFrame
            strategies: Dict of strategy_name -> optimizer
            
        Returns:
            DataFrame with comparative metrics
        """
        results = {}
        
        # Add benchmarks
        results['equal_weight'] = BenchmarkComparator.equal_weight_portfolio(returns)
        
        # Add strategies
        for name, optimizer in strategies.items():
            try:
                weights = optimizer.optimize(returns)
                results[name] = returns.dot(weights)
            except Exception as e:
                print(f"Strategy {name} failed: {e}")
        
        # Calculate metrics for each
        metrics_list = []
        for name, series in results.items():
            ann_return = series.mean() * 252
            ann_vol = series.std() * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            cum_returns = (1 + series).cumprod()
            max_dd = ((cum_returns - cum_returns.expanding().max()) / 
                     cum_returns.expanding().max()).min()
            
            metrics_list.append({
                'strategy': name,
                'annual_return': ann_return,
                'volatility': ann_vol,
                'sharpe': sharpe,
                'max_drawdown': max_dd
            })
        
        return pd.DataFrame(metrics_list)

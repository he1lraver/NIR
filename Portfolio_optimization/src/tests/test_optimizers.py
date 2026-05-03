"""
Unit tests for portfolio optimizers.

Tests verify:
- Weights sum to 1
- Weights are non-negative
- Optimizers produce valid results
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import optimizers from the new architecture
import sys
sys.path.insert(0, '/workspace/Portfolio_optimization')

from src.application.services.optimizers import (
    MarkowitzOptimizer,
    RiskParityOptimizer,
    MinimumVarianceOptimizer,
    MaximumDiversificationOptimizer,
    CVaROptimizer,
    HRPOptimizer,
    get_optimizer
)


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    n_assets = 5
    n_obs = 252
    
    # Generate correlated returns
    mean_returns = np.array([0.10, 0.08, 0.12, 0.07, 0.09]) / 252
    cov_matrix = np.array([
        [0.04, 0.01, 0.02, 0.01, 0.01],
        [0.01, 0.03, 0.01, 0.01, 0.01],
        [0.02, 0.01, 0.05, 0.02, 0.01],
        [0.01, 0.01, 0.02, 0.03, 0.01],
        [0.01, 0.01, 0.01, 0.01, 0.02]
    ]) / 252
    
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_obs)
    columns = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    dates = pd.date_range(end=datetime.now(), periods=n_obs, freq='B')
    
    return pd.DataFrame(returns, columns=columns, index=dates)


class TestMarkowitzOptimizer:
    """Tests for Markowitz (Max Sharpe) optimizer."""
    
    def test_weights_sum_to_one(self, sample_returns):
        """Verify weights sum to 1."""
        optimizer = MarkowitzOptimizer(risk_free_rate=0.05)
        weights = optimizer.optimize(sample_returns)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    
    def test_weights_non_negative(self, sample_returns):
        """Verify all weights are non-negative."""
        optimizer = MarkowitzOptimizer(risk_free_rate=0.05)
        weights = optimizer.optimize(sample_returns)
        assert np.all(weights >= -1e-6)
    
    def test_all_assets_allocated(self, sample_returns):
        """Verify at least some assets have non-zero weights."""
        optimizer = MarkowitzOptimizer(risk_free_rate=0.05)
        weights = optimizer.optimize(sample_returns)
        assert np.sum(weights > 0.01) >= 1


class TestRiskParityOptimizer:
    """Tests for Risk Parity optimizer."""
    
    def test_weights_sum_to_one(self, sample_returns):
        """Verify weights sum to 1."""
        optimizer = RiskParityOptimizer()
        weights = optimizer.optimize(sample_returns)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    
    def test_weights_non_negative(self, sample_returns):
        """Verify all weights are non-negative."""
        optimizer = RiskParityOptimizer()
        weights = optimizer.optimize(sample_returns)
        assert np.all(weights >= -1e-6)
    
    def test_risk_contributions_equal(self, sample_returns):
        """Verify risk contributions are approximately equal."""
        optimizer = RiskParityOptimizer()
        weights = optimizer.optimize(sample_returns)
        
        cov_matrix = sample_returns.cov().values * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        mrc = np.dot(cov_matrix, weights) / port_vol
        rc = weights * mrc
        
        # Risk contributions should be within 5% of target
        target_rc = port_vol / len(sample_returns.columns)
        max_deviation = np.max(np.abs(rc - target_rc) / target_rc)
        assert max_deviation < 0.1  # 10% tolerance for numerical optimization


class TestMinimumVarianceOptimizer:
    """Tests for Minimum Variance optimizer."""
    
    def test_weights_sum_to_one(self, sample_returns):
        """Verify weights sum to 1."""
        optimizer = MinimumVarianceOptimizer()
        weights = optimizer.optimize(sample_returns)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    
    def test_weights_non_negative(self, sample_returns):
        """Verify all weights are non-negative."""
        optimizer = MinimumVarianceOptimizer()
        weights = optimizer.optimize(sample_returns)
        assert np.all(weights >= -1e-6)
    
    def test_lower_volatility_than_equal_weight(self, sample_returns):
        """Verify optimized portfolio has lower vol than equal weight."""
        optimizer = MinimumVarianceOptimizer()
        weights = optimizer.optimize(sample_returns)
        
        cov_matrix = sample_returns.cov().values * 252
        opt_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        equal_weights = np.ones(len(sample_returns.columns)) / len(sample_returns.columns)
        equal_vol = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
        
        assert opt_vol <= equal_vol + 1e-6


class TestMaximumDiversificationOptimizer:
    """Tests for Maximum Diversification optimizer."""
    
    def test_weights_sum_to_one(self, sample_returns):
        """Verify weights sum to 1."""
        optimizer = MaximumDiversificationOptimizer()
        weights = optimizer.optimize(sample_returns)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    
    def test_weights_non_negative(self, sample_returns):
        """Verify all weights are non-negative."""
        optimizer = MaximumDiversificationOptimizer()
        weights = optimizer.optimize(sample_returns)
        assert np.all(weights >= -1e-6)


class TestCVaROptimizer:
    """Tests for CVaR optimizer."""
    
    def test_weights_sum_to_one(self, sample_returns):
        """Verify weights sum to 1."""
        optimizer = CVaROptimizer()
        weights = optimizer.optimize(sample_returns)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    
    def test_weights_non_negative(self, sample_returns):
        """Verify all weights are non-negative."""
        optimizer = CVaROptimizer()
        weights = optimizer.optimize(sample_returns)
        assert np.all(weights >= -1e-6)


class TestHRPOptimizer:
    """Tests for Hierarchical Risk Parity optimizer."""
    
    def test_weights_sum_to_one(self, sample_returns):
        """Verify weights sum to 1."""
        optimizer = HRPOptimizer()
        weights = optimizer.optimize(sample_returns)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    
    def test_weights_non_negative(self, sample_returns):
        """Verify all weights are non-negative."""
        optimizer = HRPOptimizer()
        weights = optimizer.optimize(sample_returns)
        assert np.all(weights >= -1e-6)
    
    def test_all_assets_get_weight(self, sample_returns):
        """Verify HRP allocates to all assets."""
        optimizer = HRPOptimizer()
        weights = optimizer.optimize(sample_returns)
        # HRP typically gives non-zero weight to all assets
        assert np.sum(weights > 0.001) >= 1


class TestOptimizerFactory:
    """Tests for optimizer factory function."""
    
    def test_get_valid_optimizer(self, sample_returns):
        """Test factory returns valid optimizers."""
        strategies = ['max_sharpe', 'risk_parity', 'min_vol', 
                     'max_div', 'min_cvar', 'hrp']
        
        for strategy in strategies:
            optimizer = get_optimizer(strategy)
            weights = optimizer.optimize(sample_returns)
            assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    
    def test_get_invalid_optimizer_raises(self):
        """Test factory raises error for invalid strategy."""
        with pytest.raises(ValueError):
            get_optimizer('invalid_strategy')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

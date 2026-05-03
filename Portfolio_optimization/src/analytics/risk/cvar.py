"""
Advanced risk analytics module.

Implements canonical algorithms from academic literature:
- Rockafellar-Uryasev CVaR optimization (2000)
- Ledoit-Wolf shrinkage covariance estimation
- Numerical stability diagnostics
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional


class RiskAnalytics:
    """
    Advanced risk calculation utilities.
    """
    
    @staticmethod
    def calculate_var_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate historical Value at Risk.
        
        Args:
            returns: Array of returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR as a positive number representing potential loss
        """
        percentile = (1 - confidence) * 100
        return -np.percentile(returns, percentile)
    
    @staticmethod
    def calculate_cvar_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate historical Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Array of returns
            confidence: Confidence level
            
        Returns:
            CVaR as expected loss beyond VaR threshold
        """
        var = RiskAnalytics.calculate_var_historical(returns, confidence)
        tail_losses = returns[returns <= -var]
        if len(tail_losses) == 0:
            return var
        return -tail_losses.mean()
    
    @staticmethod
    def calculate_cvar_cornish_fisher(returns: np.ndarray, 
                                       confidence: float = 0.95) -> float:
        """
        Calculate CVaR with Cornish-Fisher adjustment for non-normal distributions.
        
        Accounts for skewness and kurtosis in the return distribution.
        
        Args:
            returns: Array of returns
            confidence: Confidence level
            
        Returns:
            Adjusted CVaR
        """
        z = stats.norm.ppf(1 - confidence)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        # Cornish-Fisher expansion
        z_cf = (z + 
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * (skew**2) / 36)
        
        mean = np.mean(returns)
        std = np.std(returns)
        var_cf = -(mean + z_cf * std)
        
        # Approximate CVaR from CF-VaR
        tail_losses = returns[returns <= -var_cf]
        if len(tail_losses) == 0:
            return var_cf
        return -tail_losses.mean()
    
    @staticmethod
    def calculate_expected_shortfall(returns: np.ndarray, 
                                      confidence: float = 0.95) -> float:
        """Alias for CVaR calculation."""
        return RiskAnalytics.calculate_cvar_historical(returns, confidence)
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from returns series.
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown as a negative number
        """
        cum_returns = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return drawdowns.min()
    
    @staticmethod
    def calculate_calmar_ratio(annual_return: float, 
                                max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (return / max_drawdown).
        
        Args:
            annual_return: Annualized return
            max_drawdown: Maximum drawdown (as positive number)
            
        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0
        return annual_return / max_drawdown
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, 
                                 risk_free_rate: float = 0.0,
                                 annualize: bool = True) -> float:
        """
        Calculate Sortino ratio (excess return / downside deviation).
        
        Only penalizes downside volatility.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (daily if not annualized)
            annualize: Whether to annualize the result
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_std == 0:
            return np.inf
        
        mean_excess = np.mean(excess_returns)
        sortino = mean_excess / downside_std
        
        if annualize:
            sortino *= np.sqrt(252)
        
        return sortino


class CovarianceEstimator:
    """
    Advanced covariance matrix estimation techniques.
    """
    
    @staticmethod
    def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate covariance using Ledoit-Wolf shrinkage estimator.
        
        Shrinks sample covariance toward structured target (scaled identity),
        improving numerical stability for ill-conditioned matrices.
        
        Reference: Ledoit & Wolf, "Honey, I Shrunk the Sample Covariance" (2004)
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Shrunken covariance matrix (annualized)
        """
        from sklearn.covariance import LedoitWolf
        
        lw = LedoitWolf()
        lw.fit(returns.values)
        cov_matrix = lw.covariance_
        
        # Annualize
        return cov_matrix * 252
    
    @staticmethod
    def check_positive_semi_definite(matrix: np.ndarray, 
                                      tolerance: float = 1e-10) -> Tuple[bool, np.ndarray]:
        """
        Check if matrix is positive semi-definite.
        
        Args:
            matrix: Square matrix to check
            tolerance: Tolerance for negative eigenvalues
            
        Returns:
            Tuple of (is_psd, eigenvalues)
        """
        eigenvalues = np.linalg.eigvalsh(matrix)
        is_psd = np.all(eigenvalues >= -tolerance)
        return is_psd, eigenvalues
    
    @staticmethod
    def condition_number(matrix: np.ndarray) -> float:
        """
        Calculate condition number of a matrix.
        
        High condition numbers (>10^10) indicate numerical instability.
        
        Args:
            matrix: Square matrix
            
        Returns:
            Condition number
        """
        return np.linalg.cond(matrix)
    
    @staticmethod
    def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
        """
        Find nearest positive definite matrix using eigenvalue clipping.
        
        Args:
            matrix: Square symmetric matrix
            
        Returns:
            Nearest positive definite matrix
        """
        # Ensure symmetry
        matrix = (matrix + matrix.T) / 2
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Clip negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        # Reconstruct
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


class DiagnosticsReport:
    """
    Generate comprehensive diagnostics for portfolio optimization.
    """
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.cov_sample = returns.cov().values * 252
        self.n_assets = len(returns.columns)
        self.n_obs = len(returns)
    
    def generate(self) -> Dict:
        """Generate full diagnostics report."""
        report = {
            'data_quality': self._check_data_quality(),
            'covariance_analysis': self._analyze_covariance(),
            'numerical_stability': self._check_numerical_stability(),
            'recommendations': []
        }
        
        # Add recommendations based on findings
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _check_data_quality(self) -> Dict:
        """Check data quality metrics."""
        missing_pct = self.returns.isnull().sum().sum() / self.returns.size * 100
        
        return {
            'n_assets': self.n_assets,
            'n_observations': self.n_obs,
            'missing_data_pct': missing_pct,
            'min_obs_per_asset': int(self.returns.notnull().sum().min()),
            'adequate_data': self.n_obs >= 252 and missing_pct < 5
        }
    
    def _analyze_covariance(self) -> Dict:
        """Analyze covariance matrix properties."""
        is_psd, eigenvalues = CovarianceEstimator.check_positive_semi_definite(
            self.cov_sample
        )
        
        return {
            'is_positive_semi_definite': is_psd,
            'min_eigenvalue': float(np.min(eigenvalues)),
            'max_eigenvalue': float(np.max(eigenvalues)),
            'eigenvalue_spread': float(np.max(eigenvalues) - np.min(eigenvalues)),
            'rank': int(np.linalg.matrix_rank(self.cov_sample))
        }
    
    def _check_numerical_stability(self) -> Dict:
        """Check numerical stability indicators."""
        cond_num = CovarianceEstimator.condition_number(self.cov_sample)
        
        return {
            'condition_number': float(cond_num),
            'well_conditioned': cond_num < 1e10,
            'moderately_ill_conditioned': 1e10 <= cond_num < 1e15,
            'severely_ill_conditioned': cond_num >= 1e15
        }
    
    def _generate_recommendations(self, report: Dict) -> list:
        """Generate recommendations based on diagnostics."""
        recommendations = []
        
        if not report['data_quality']['adequate_data']:
            recommendations.append(
                "Insufficient data: consider extending the lookback period"
            )
        
        if not report['covariance_analysis']['is_positive_semi_definite']:
            recommendations.append(
                "Covariance matrix not PSD: use Ledoit-Wolf shrinkage or "
                "nearest positive definite correction"
            )
        
        if not report['numerical_stability']['well_conditioned']:
            recommendations.append(
                "High condition number: consider regularization, shrinkage, "
                "or reducing number of assets"
            )
        
        if self.n_assets > self.n_obs / 10:
            recommendations.append(
                f"Too many assets ({self.n_assets}) relative to observations "
                f"({self.n_obs}): risk of overfitting"
            )
        
        return recommendations

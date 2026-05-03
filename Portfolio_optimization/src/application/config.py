"""
Configuration management for reproducible experiments.

Supports YAML configuration files for experiment parameters.
"""
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import date
import json


@dataclass
class ExperimentConfig:
    """
    Configuration for a portfolio optimization experiment.
    
    Enables reproducible research by capturing all parameters.
    """
    # Market data settings
    market: str = "moex"
    tickers: List[str] = field(default_factory=list)
    start_date: str = ""  # YYYY-MM-DD or "dynamic" for rolling
    end_date: str = ""    # YYYY-MM-DD or "dynamic"
    lookback_days: int = 756  # ~3 years
    
    # Optimization settings
    strategy: str = "max_sharpe"
    risk_free_rate: float = 0.16
    allow_short_selling: bool = False
    
    # Backtesting settings  
    train_window: int = 252
    rebalance_period: int = 21
    min_train_samples: int = 126
    
    # Risk settings
    use_shrinkage_covariance: bool = False
    cvar_confidence: float = 0.95
    
    # Output settings
    save_results: bool = True
    verbose: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    def to_json(self) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(asdict(self), indent=2)
    
    def validate(self) -> tuple[bool, str]:
        """Validate configuration parameters."""
        if not self.tickers and self.start_date != "dynamic":
            return False, "No tickers specified"
        
        if self.risk_free_rate < 0 or self.risk_free_rate > 1:
            return False, "Risk-free rate must be between 0 and 1"
        
        if self.cvar_confidence <= 0 or self.cvar_confidence >= 1:
            return False, "CVaR confidence must be between 0 and 1"
        
        valid_strategies = ['max_sharpe', 'risk_parity', 'min_vol', 
                           'max_div', 'min_cvar', 'hrp']
        if self.strategy not in valid_strategies:
            return False, f"Strategy must be one of {valid_strategies}"
        
        return True, "OK"


# Default configurations for common experiments
DEFAULT_CONFIGS = {
    'conservative': ExperimentConfig(
        strategy='min_vol',
        risk_free_rate=0.10,
        use_shrinkage_covariance=True
    ),
    'balanced': ExperimentConfig(
        strategy='risk_parity',
        risk_free_rate=0.16,
        rebalance_period=42
    ),
    'aggressive': ExperimentConfig(
        strategy='max_sharpe',
        risk_free_rate=0.20,
        rebalance_period=21
    ),
    'diversified': ExperimentConfig(
        strategy='max_div',
        risk_free_rate=0.16
    ),
    'tail_risk_aware': ExperimentConfig(
        strategy='min_cvar',
        risk_free_rate=0.16,
        cvar_confidence=0.99
    ),
    'hierarchical': ExperimentConfig(
        strategy='hrp',
        risk_free_rate=0.16,
        use_shrinkage_covariance=False
    )
}


def get_config(preset: str) -> ExperimentConfig:
    """Get a predefined configuration preset."""
    if preset not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. "
                        f"Available: {list(DEFAULT_CONFIGS.keys())}")
    return DEFAULT_CONFIGS[preset]

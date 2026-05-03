"""
MOEX (Moscow Exchange) data provider implementation.

This module adapts the existing MoexDataLoader to the new architecture.
"""
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple


class MoexDataProvider:
    """
    Data provider for Moscow Exchange using apimoex API.
    
    Implements the DataProvider protocol from domain.interfaces.
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
    
    def fetch_prices(self, tickers: list, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical prices for given tickers.
        
        Args:
            tickers: List of ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with prices or None if failed
        """
        result = self._fetch_all(tickers, start, end)
        if result is None:
            return None
        prices, _ = result
        return prices
    
    def fetch_returns(self, tickers: list, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical returns for given tickers.
        
        Args:
            tickers: List of ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with returns or None if failed
        """
        result = self._fetch_all(tickers, start, end)
        if result is None:
            return None
        _, returns = result
        return returns
    
    def _fetch_sync(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch data for a single ticker synchronously."""
        try:
            import apimoex
            import requests
            
            with requests.Session() as session:
                data = apimoex.get_board_history(
                    session,
                    security=ticker,
                    board='TQBR',
                    start=start,
                    end=end,
                    columns=('TRADEDATE', 'CLOSE')
                )
                
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
                df.set_index('TRADEDATE', inplace=True)
                df.rename(columns={'CLOSE': ticker}, inplace=True)
                return df
                
        except Exception:
            return pd.DataFrame()
    
    def _fetch_all(self, tickers: list, start: str, end: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Fetch and process data for all tickers.
        
        Returns:
            Tuple of (prices, returns) or None if failed
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        valid_dfs = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_sync, ticker, start, end): ticker
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                df = future.result()
                if not df.empty:
                    valid_dfs.append(df)
        
        if not valid_dfs:
            return None
        
        # Combine all tickers
        prices = pd.concat(valid_dfs, axis=1).sort_index()
        prices = prices.ffill().bfill()
        
        # Adjust for splits
        for col in prices.columns:
            ratios = prices[col] / prices[col].shift(1)
            splits = ratios[ratios < 0.4]
            
            for date, ratio in splits.items():
                split_factor = round(1 / ratio)
                if split_factor > 1:
                    prices.loc[prices.index < date, col] /= split_factor
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Filter out low-volatility assets
        valid_cols = returns.columns[returns.std() > 0.0001]
        
        return prices[valid_cols], returns[valid_cols]

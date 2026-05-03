import pandas as pd
import apimoex
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class MoexDataLoader:
    def __init__(self):
        # Количество одновременных соединений
        self.max_workers = 10 

    def _fetch_sync(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Синхронный вызов к API мосбиржи (создаем отдельную сессию для потокобезопасности)"""
        try:
            with requests.Session() as session:
                data = apimoex.get_board_history(
                    session, security=ticker, board='TQBR',
                    start=start, end=end, columns=('TRADEDATE', 'CLOSE')
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

    def fetch_all(self, tickers: list, start_date: datetime, end_date: datetime):
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        valid_dfs = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_sync, ticker, start_str, end_str): ticker 
                for ticker in tickers
            }
            for future in as_completed(future_to_ticker):
                df = future.result()
                if not df.empty:
                    valid_dfs.append(df)
        
        if not valid_dfs:
            return None
            
        prices = pd.concat(valid_dfs, axis=1).sort_index()
        prices = prices.ffill().bfill()
        
        # === АЛГОРИТМ АВТО-КОРРЕКЦИИ СПЛИТОВ (Adjusted Prices) ===
        for col in prices.columns:
            # Ищем отношение цены сегодня к цене вчера
            ratios = prices[col] / prices[col].shift(1)
            # Если цена упала более чем на 60% за день - это сплит (как у TRNFP)
            splits = ratios[ratios < 0.4] 
            
            for date, ratio in splits.items():
                split_factor = round(1 / ratio) # Определяем коэффициент (например, 100)
                if split_factor > 1:
                    # Делим все исторические цены ДО сплита на коэффициент
                    prices.loc[prices.index < date, col] /= split_factor
        # ==========================================================

        returns = prices.pct_change().dropna()
        valid_cols = returns.columns[returns.std() > 0.0001]
        
        return prices[valid_cols], returns[valid_cols]
"""
Модуль загрузки финансовых данных с Московской Биржи (MOEX).

Использует официальный API Мосбиржи через библиотеку apimoex.
Реализует многопоточную загрузку для ускорения получения данных
и автоматическую коррекцию цен на сплиты (дробления акций).

Author: [Ваше ФИО]
Version: 1.0
"""
import pandas as pd
import apimoex
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


class MoexDataLoader:
    """
    Класс для загрузки исторических данных о ценах акций с Московской Биржи.
    
    Attributes:
        max_workers: Количество потоков для параллельной загрузки
    """
    
    def __init__(self):
        self.max_workers = 10

    def _fetch_sync(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Синхронный запрос к API MOEX для одного тикера.
        
        Args:
            ticker: Биржевой тикер (например, 'SBER')
            start: Дата начала в формате YYYY-MM-DD
            end: Дата окончания в формате YYYY-MM-DD
            
        Returns:
            DataFrame с колонками TRADEDATE (индекс) и CLOSE (цена закрытия)
        """
        try:
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

    def fetch_all(self, tickers: list, start_date: datetime, end_date: datetime):
        """
        Загрузка данных по всем тикерам с последующей обработкой.
        
        Args:
            tickers: Список тикеров для загрузки
            start_date: Дата начала анализа
            end_date: Дата окончания анализа
            
        Returns:
            Кортеж (prices, returns):
                - prices: DataFrame с ценами закрытия (с коррекцией на сплиты)
                - returns: DataFrame с дневными доходностями
                - None при ошибке загрузки
        """
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
        
        for col in prices.columns:
            ratios = prices[col] / prices[col].shift(1)
            splits = ratios[ratios < 0.4]
            
            for date, ratio in splits.items():
                split_factor = round(1 / ratio)
                if split_factor > 1:
                    prices.loc[prices.index < date, col] /= split_factor
        
        returns = prices.pct_change().dropna()
        valid_cols = returns.columns[returns.std() > 0.0001]
        
        return prices[valid_cols], returns[valid_cols]

"""
Модуль математических моделей оптимизации портфеля.

Реализует классические и современные методы квантитативного анализа:
- Modern Portfolio Theory (Markowitz, 1952)
- Risk Parity (Qian, 2005; All Weather by Dalio)
- Hierarchical Risk Parity (Lopez de Prado, 2016)
- Conditional Value at Risk (Rockafellar & Uryasev, 2000)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


class PortfolioMath:
    """
    Класс для расчёта метрик и оптимизации весов портфеля ценных бумаг.
    
    Attributes:
        returns: DataFrame с дневными доходностями активов
        risk_free_rate: Безрисковая ставка (дневная)
        cov_matrix: Ковариационная матрица (годовая)
        mean_returns: Вектор ожидаемых доходностей (годовой)
        vols: Волатильности активов (годовые)
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.05):
        self.returns = returns
        self.rfr = risk_free_rate / 252
        self.cov_matrix = returns.cov() * 252
        self.mean_returns = returns.mean() * 252
        self.vols = returns.std() * np.sqrt(252)

    def calc_metrics(self, weights: np.ndarray) -> dict:
        """
        Расчёт ключевых метрик портфеля.
        
        Args:
            weights: Вектор весов активов
            
        Returns:
            Dictionary с метриками:
                - return: Ожидаемая годовая доходность
                - volatility: Годовая волатильность (стандартное отклонение)
                - sharpe: Коэффициент Шарпа (отношение избыточной доходности к риску)
                - var_95: Value at Risk (95%, годовой, Cornish-Fisher корректировка)
                - cvar_95: Conditional VaR / Expected Shortfall (95%, годовой)
                - max_drawdown: Максимальная историческая просадка
                - daily_returns: Серия дневных доходностей портфеля
                - drawdowns: Серия просадок портфеля
        """
        # Портфельная доходность и волатильность
        port_return = np.sum(self.mean_returns * weights)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Коэффициент Шарпа: (E[R_p] - R_f) / σ_p
        sharpe = (port_return - (self.rfr * 252)) / port_volatility if port_volatility > 0 else 0
        
        # Дневные доходности портфеля для расчёта хвостовых метрик
        daily_returns = self.returns.dot(weights)
        
        # Cornish-Fisher VaR & CVaR — корректировка на асимметрию и эксцесс
        z = stats.norm.ppf(0.05)  # Квантиль нормального распределения (5%)
        skew = stats.skew(daily_returns)  # Асимметрия распределения
        kurt = stats.kurtosis(daily_returns)  # Эксцесс (избыточный)
        
        # Формула Cornish-Fisher для квантиля не-нормального распределения
        z_cf = (z + 
                (z**2 - 1) * skew / 6 + 
                (z**3 - 3*z) * kurt / 24 - 
                (2*z**3 - 5*z) * (skew**2) / 36)
        
        var_95 = -(daily_returns.mean() + z_cf * daily_returns.std())
        
        # CVaR — среднее значение потерь за порогом VaR
        tail_losses = daily_returns[daily_returns <= -var_95]
        cvar_95 = -tail_losses.mean() if len(tail_losses) > 0 else var_95
        
        # Расчёт максимальной просадки через кумулятивную доходность
        cum_returns = (1 + daily_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        
        return {
            'return': port_return,
            'volatility': port_volatility,
            'sharpe': sharpe,
            'var_95': var_95 * np.sqrt(252),
            'cvar_95': cvar_95 * np.sqrt(252),
            'max_drawdown': drawdowns.min(),
            'daily_returns': daily_returns,
            'drawdowns': drawdowns
        }

    def optimize_sharpe(self) -> np.ndarray:
        """
        Оптимизация по критерию максимального коэффициента Шарпа (Modern Portfolio Theory).
        
        Математическая постановка задачи:
            max_w: Sharpe(w) = (E[R_p] - R_f) / σ_p
            при условиях:
                Σw_i = 1 (полное инвестирование)
                w_i ≥ 0 (отсутствие коротких позиций)
        
        Returns:
            Вектор оптимальных весов портфеля
        """
        num_assets = len(self.returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
        res = minimize(
            fun=lambda w: -self.calc_metrics(w)['sharpe'],
            x0=num_assets * [1./num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return res.x

    def optimize_risk_parity(self) -> np.ndarray:
        """
        Стратегия Risk Parity (паритет рисков) — модель All Weather (Рэй Далио).
        
        Идея: каждый актив вносит одинаковый вклад в общий риск портфеля.
        
        Математически:
            min_w: Σ(RC_i - RC_target)²
            где RC_i = w_i × MRC_i — marginal contribution to risk
                 RC_target = σ_p / N
        
        Returns:
            Вектор весов с равным вкладом в риск
        """
        num_assets = len(self.returns.columns)
        
        def risk_budget_objective(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            mrc = np.dot(self.cov_matrix, weights) / port_vol  # Предельный вклад в риск
            rc = weights * mrc  # Фактический вклад каждого актива
            target_rc = port_vol / num_assets
            return np.sum(np.square(rc - target_rc))
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
        res = minimize(
            fun=risk_budget_objective,
            x0=num_assets * [1./num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return res.x

    def optimize_min_vol(self) -> np.ndarray:
        """
        Портфель глобальной минимальной волатильности (Global Minimum Variance Portfolio).
        
        Задача квадратичного программирования:
            min_w: σ_p² = w'Σw
            при условиях:
                Σw_i = 1
                w_i ≥ 0
        
        Returns:
            Вектор весов с минимальной волатильностью
        """
        num_assets = len(self.returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
        res = minimize(
            fun=lambda w: np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w))),
            x0=num_assets * [1./num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return res.x

    def optimize_max_div(self) -> np.ndarray:
        """
        Портфель максимальной диверсификации (Maximum Diversification Portfolio).
        
        Коэффициент диверсификации:
            DR(w) = (Σw_iσ_i) / σ_p
        
        Максимизация DR означает, что портфель получает максимум 
        взвешенной волатильности при минимуме фактического риска.
        
        Returns:
            Вектор весов с максимальным коэффициентом диверсификации
        """
        num_assets = len(self.returns.columns)
        
        def calc_div_ratio(w):
            port_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            weighted_vols = np.dot(w, self.vols)
            return -(weighted_vols / port_vol)  # Минимизируем отрицательный коэффициент
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
        res = minimize(
            fun=calc_div_ratio,
            x0=num_assets * [1./num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return res.x

    def optimize_min_cvar(self) -> np.ndarray:
        """
        Минимизация хвостового риска (Conditional Value at Risk).
        
        CVaR (Expected Shortfall) — математическое ожидание потерь
        при превышении порога VaR. Более консервативная метрика, чем VaR.
        
        Returns:
            Вектор весов с минимальным CVaR
        """
        num_assets = len(self.returns.columns)
        
        def cvar_objective(w):
            return self.calc_metrics(w)['cvar_95']
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
        res = minimize(
            fun=cvar_objective,
            x0=num_assets * [1./num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return res.x

    def optimize_hrp(self) -> np.ndarray:
        """
        Иерархический паритет рисков (Hierarchical Risk Parity, Lopez de Prado, 2016).
        
        Алгоритм использует кластеризацию иерархической структуры активов
        для распределения веса без обращения ковариационной матрицы.
        
        Шаги:
        1. Построение матрицы расстояний на основе корреляций
        2. Иерархическая кластеризация (single linkage)
        3. Рекурсивное бисекционное распределение веса
        
        Returns:
            Вектор весов HRP-портфеля
        """
        corr = self.returns.corr().values
        
        # Преобразование корреляции в расстояние: d = sqrt((1 - corr) / 2)
        dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
        condensed_dist = squareform(dist, checks=False)
        
        # Иерархическая кластеризация
        link = sch.linkage(condensed_dist, method='single')
        sort_ix = sch.leaves_list(link)  # Индексы отсортированных листьев
        
        def get_rec_bipart(cov, sort_ix):
            """Рекурсивное распределение веса по бинарным кластерам."""
            w = pd.Series(1.0, index=sort_ix)
            clusters = [sort_ix]
            
            while len(clusters) > 0:
                # Разбиение кластеров на подкластеры
                clusters = [
                    cluster[j:k] 
                    for cluster in clusters 
                    for j, k in ((0, len(cluster)//2), (len(cluster)//2, len(cluster))) 
                    if len(cluster) > 1
                ]
                
                # Попарное объединение и перераспределение веса
                for i in range(0, len(clusters), 2):
                    c_1 = clusters[i]
                    c_2 = clusters[i + 1]
                    
                    # Дисперсия подкластеров
                    var_c1 = np.dot(w[c_1].T, np.dot(cov[np.ix_(c_1, c_1)], w[c_1]))
                    var_c2 = np.dot(w[c_2].T, np.dot(cov[np.ix_(c_2, c_2)], w[c_2]))
                    
                    # Альфа — доля веса первого кластера (обратно пропорциональна риску)
                    alpha = 1 - var_c1 / (var_c1 + var_c2)
                    w[c_1] *= alpha
                    w[c_2] *= (1 - alpha)
            
            return w
        
        weights = get_rec_bipart(self.cov_matrix.values, sort_ix)
        
        # Возвращение весов в исходном порядке тикеров
        final_weights = np.zeros(len(self.returns.columns))
        for i, idx in enumerate(sort_ix):
            final_weights[idx] = weights.iloc[i] if isinstance(weights.iloc[i], (int, float)) else weights[idx]
        
        return final_weights / np.sum(final_weights)
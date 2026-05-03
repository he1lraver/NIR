import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

class PortfolioMath:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.05):
        self.returns = returns
        self.rfr = risk_free_rate / 252
        self.cov_matrix = returns.cov() * 252
        self.mean_returns = returns.mean() * 252
        self.vols = returns.std() * np.sqrt(252)

    def calc_metrics(self, weights: np.ndarray) -> dict:
        port_return = np.sum(self.mean_returns * weights)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (port_return - (self.rfr*252)) / port_volatility if port_volatility > 0 else 0
        
        daily_returns = self.returns.dot(weights)
        
        # Cornish-Fisher VaR & CVaR
        z = stats.norm.ppf(0.05)
        skew = stats.skew(daily_returns)
        kurt = stats.kurtosis(daily_returns)
        z_cf = z + (z**2 - 1)*skew/6 + (z**3 - 3*z)*kurt/24 - (2*z**3 - 5*z)*(skew**2)/36
        
        var_95 = -(daily_returns.mean() + z_cf * daily_returns.std())
        cvar_95 = -daily_returns[daily_returns <= -var_95].mean()
        if pd.isna(cvar_95): cvar_95 = var_95
        
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

    # 1. Максимальный Шарп (Классика MPT Марковица)
    def optimize_sharpe(self):
        num_assets = len(self.returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        res = minimize(lambda w: -self.calc_metrics(w)['sharpe'], 
                       num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

    # 2. Паритет риска (Risk Parity - All Weather Dalio)
    def optimize_risk_parity(self):
        num_assets = len(self.returns.columns)
        def risk_budget_objective(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            mrc = np.dot(self.cov_matrix, weights) / port_vol
            rc = weights * mrc
            return np.sum(np.square(rc - (port_vol / num_assets)))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        res = minimize(risk_budget_objective, num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

    # 3. Глобальный минимум дисперсии (Global Minimum Volatility)
    def optimize_min_vol(self):
        num_assets = len(self.returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        res = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w))), 
                       num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

    # 4. Максимальная диверсификация (Maximum Diversification Ratio)
    def optimize_max_div(self):
        num_assets = len(self.returns.columns)
        def calc_div_ratio(w):
            port_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            weighted_vols = np.dot(w, self.vols)
            return -(weighted_vols / port_vol) # Минимизируем отрицательный коэф.
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        res = minimize(calc_div_ratio, num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

    # 5. Минимизация хвостовых рисков (Min CVaR - защита от кризисов)
    def optimize_min_cvar(self):
        num_assets = len(self.returns.columns)
        def cvar_objective(w):
            return self.calc_metrics(w)['cvar_95']
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        res = minimize(cvar_objective, num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

    # 6. Иерархический паритет риска (Machine Learning - HRP)
    def optimize_hrp(self):
        corr = self.returns.corr().values
        dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1)) # Дистанционная матрица
        condensed_dist = squareform(dist, checks=False)
        link = sch.linkage(condensed_dist, 'single') # Кластеризация
        sort_ix = sch.leaves_list(link) # Сортировка листьев
        
        # Рекурсивное распределение риска по кластерам
        def get_rec_bipart(cov, sort_ix):
            w = pd.Series(1.0, index=sort_ix)
            c_items = [sort_ix]
            while len(c_items) > 0:
                c_items = [i[j:k] for i in c_items for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i) > 1]
                for i in range(0, len(c_items), 2):
                    c_1 = c_items[i]
                    c_2 = c_items[i+1]
                    c_1_v = np.dot(w[c_1].T, np.dot(cov[np.ix_(c_1, c_1)], w[c_1]))
                    c_2_v = np.dot(w[c_2].T, np.dot(cov[np.ix_(c_2, c_2)], w[c_2]))
                    alpha = 1 - c_1_v / (c_1_v + c_2_v)
                    w[c_1] *= alpha
                    w[c_2] *= (1 - alpha)
            return w

        weights = get_rec_bipart(self.cov_matrix.values, sort_ix)
        # Возвращаем веса в исходном порядке тикеров
        final_weights = np.zeros(len(self.returns.columns))
        for i, idx in enumerate(sort_ix):
            final_weights[idx] = weights[idx]
        return final_weights / np.sum(final_weights)
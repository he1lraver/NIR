import streamlit as st
import pandas as pd
import numpy as np
import requests
import apimoex
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats
from statsmodels.stats.stattools import jarque_bera
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Настройки страницы
st.set_page_config(
    page_title="AI Portfolio Optimizer - MOEX",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS для улучшения стилей
st.markdown("""
<style>
    /* Основные стили */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Карточки */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -4px rgba(0, 0, 0, 0.15);
    }
    
    /* Сайдбар */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d3748 0%, #4a5568 100%);
    }
    
    .sidebar-header {
        color: white !important;
        font-weight: 600;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    
    /* Кнопки */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Вкладки */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background: white;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-weight: 500;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #667eea;
        color: white;
        border-color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: #667eea !important;
    }
    
    /* Прогресс бар */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Селекторы */
    .stSelectbox, .stTextArea, .stDateInput, .stSlider {
        margin-bottom: 1rem;
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    /* Метрики */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #718096;
    }
    
    /* Уведомления */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    
    /* Адаптивность */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class PortfolioOptimizer:
    def __init__(self):
        self.prices = None
        self.returns = None
        self.optimal_weights = None
        
    def load_data(self, tickers, start_date, end_date):
        """Загрузка данных с MOEX с расширенной обработкой ошибок"""
        data_frames = []
        successful_tickers = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_container = st.container()
        
        with status_container:
            st.info("🔄 Начинаем загрузку данных...")
            
            for i, ticker in enumerate(tickers):
                status_text.text(f"📥 Загружаем {ticker}... ({i+1}/{len(tickers)})")
                try:
                    with requests.Session() as session:
                        data = apimoex.get_board_history(
                            session,
                            security=ticker,
                            board='TQBR',
                            market="shares",
                            engine="stock",
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'),
                            columns=('TRADEDATE', 'CLOSE', 'VOLUME')
                        )
                    
                    if data and len(data) > 10:
                        df = pd.DataFrame(data)
                        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
                        df.set_index('TRADEDATE', inplace=True)
                        df = df[['CLOSE']].rename(columns={'CLOSE': ticker})
                        data_frames.append(df)
                        successful_tickers.append(ticker)
                        st.success(f"✅ {ticker} успешно загружен")
                    else:
                        st.warning(f"⚠️ Недостаточно данных для {ticker}")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка загрузки {ticker}: {str(e)}")
                    
                progress_bar.progress((i + 1) / len(tickers))
            
        status_text.empty()
        progress_bar.empty()
        
        if data_frames:
            self.prices = pd.concat(data_frames, axis=1).sort_index()
            self.prices = self.prices.ffill().bfill()
            self.returns = self.prices.pct_change().dropna()
            
            # Удаляем активы с нулевой волатильностью
            zero_vol_tickers = self.returns.columns[self.returns.std() == 0]
            if len(zero_vol_tickers) > 0:
                st.warning(f"⚠️ Удалены активы с нулевой волатильностью: {list(zero_vol_tickers)}")
                self.returns = self.returns.drop(columns=zero_vol_tickers)
                self.prices = self.prices.drop(columns=zero_vol_tickers)
            
            if len(self.returns.columns) == 0:
                st.error("❌ Нет активов с достаточной волатильностью для оптимизации")
                return False
            
            with status_container:
                st.success(f"🎉 Данные успешно загружены! Обработано {len(successful_tickers)} из {len(tickers)} активов")
            return True
        
        with status_container:
            st.error("😔 Не удалось загрузить данные для выбранных тикеров")
        return False
    
    def portfolio_metrics(self, weights):
        """Расчет метрик портфеля с исправлениями"""
        portfolio_returns = self.returns.dot(weights)
        
        # Годовые показатели
        annual_return = np.mean(portfolio_returns) * 252
        annual_volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Защита от нулевой волатильности
        if annual_volatility == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        
        # Просадки
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Коэффициент Сортино с защитой
        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
        else:
            downside_deviation = 0
            
        if downside_deviation > 0:
            sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation
        else:
            sortino_ratio = 0
        
        # VaR и CVaR
        if len(portfolio_returns) > 0:
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        else:
            var_95 = 0
            cvar_95 = 0
        
        # Бета (относительно среднерыночного портфеля)
        if len(self.returns.columns) > 1:
            market_returns = self.returns.mean(axis=1)  # Среднерыночная доходность
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 1
        else:
            beta = 1
            
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'beta': beta,
            'cumulative_returns': cumulative_returns,
            'drawdown': drawdown
        }
    
    def optimize_portfolio(self, risk_free_rate=0.05, method='sharpe'):
        """Оптимизация портфеля с улучшенной стабильностью"""
        self.risk_free_rate = risk_free_rate
        
        def sharpe_ratio(weights):
            metrics = self.portfolio_metrics(weights)
            return -metrics['sharpe_ratio']  # Минимизируем отрицательный Шарп
        
        def volatility(weights):
            metrics = self.portfolio_metrics(weights)
            return metrics['annual_volatility']
        
        # Ограничения и границы
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(len(self.returns.columns)))
        
        # Несколько разных начальных точек для лучшей оптимизации
        best_result = None
        best_value = np.inf
        
        with st.spinner("🔍 Ищем оптимальное распределение активов..."):
            for _ in range(5):  # Пробуем несколько начальных точек
                # Начальные веса с небольшим случайным шумом
                init_weights = np.random.dirichlet(np.ones(len(self.returns.columns)))
                
                # Выбор целевой функции
                if method == 'sharpe':
                    objective = sharpe_ratio
                elif method == 'min_vol':
                    objective = volatility
                
                try:
                    result = minimize(
                        objective, init_weights, 
                        method='SLSQP', bounds=bounds, constraints=constraints,
                        options={'ftol': 1e-9, 'maxiter': 1000}
                    )
                    
                    if result.success and result.fun < best_value:
                        best_result = result
                        best_value = result.fun
                        
                except Exception as e:
                    continue
        
        if best_result is not None and best_result.success:
            self.optimal_weights = best_result.x
            # Нормализуем веса (на случай небольших отклонений)
            self.optimal_weights = self.optimal_weights / np.sum(self.optimal_weights)
            return True
        else:
            st.error("❌ Оптимизация не удалась. Попробуйте изменить параметры или тикеры.")
            return False

def main():
    # Заголовок с градиентом
    st.markdown('<h1 class="main-header">🎯 AI Portfolio Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Интеллектуальная оптимизация портфеля акций Московской биржи</p>', unsafe_allow_html=True)
    
    # Инициализация оптимизатора
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = PortfolioOptimizer()
    
    # Сайдбар с настройками
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Настройки портфеля</div>', unsafe_allow_html=True)
        
        # Выбор тикеров
        st.subheader("📊 Выбор активов")
        popular_tickers = {
            "Базовый портфель": "SBER, GAZP, LKOH, GMKN, TATN",
            "Голубые фишки": "SBER, GAZP, LKOH, GMKN, ROSN, TATN, MTSS, NLMK",
            "Дивидендные": "SBER, GAZP, LKOH, GMKN, TATN, MTSS, MGNT, PHOR",
            "Ростовые": "YNDX, OZON, TCSG, POSI, PLZL"
        }
        
        portfolio_type = st.selectbox("**Тип портфеля**", list(popular_tickers.keys()))
        custom_tickers = st.text_area(
            "**Или введите свои тикеры**", 
            value=popular_tickers[portfolio_type],
            help="Введите тикеры через запятую, например: SBER, GAZP, LKOH",
            height=100
        )
        tickers = [ticker.strip().upper() for ticker in custom_tickers.split(',') if ticker.strip()]
        
        # Период анализа
        st.subheader("📅 Период анализа")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*3)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("**Начало**", start_date)
        with col2:
            end_date = st.date_input("**Конец**", end_date)
        
        # Параметры оптимизации
        st.subheader("🎯 Параметры оптимизации")
        risk_free_rate = st.slider("**Безрисковая ставка (%)**", 0.0, 15.0, 5.0, help="Годовая безрисковая ставка в процентах") / 100
        optimization_method = st.selectbox(
            "**Метод оптимизации**", 
            ["sharpe", "min_vol"],
            format_func=lambda x: "📈 Максимизация Шарпа" if x == "sharpe" else "🛡️ Минимизация волатильности"
        )
        
        # Разделитель
        st.markdown("---")
        
        # Кнопка расчета
        if st.button("🚀 Рассчитать оптимальный портфель", type="primary", use_container_width=True):
            with st.spinner("⏳ Загрузка данных и оптимизация..."):
                if st.session_state.optimizer.load_data(tickers, start_date, end_date):
                    # Проверка качества данных
                    if len(st.session_state.optimizer.returns.columns) < 2:
                        st.error("❌ Недостаточно активов для диверсификации. Добавьте больше тикеров.")
                    elif st.session_state.optimizer.returns.isna().any().any():
                        st.warning("⚠️ В данных есть пропуски. Проводим очистку...")
                        st.session_state.optimizer.returns = st.session_state.optimizer.returns.dropna()
                    
                    if st.session_state.optimizer.optimize_portfolio(risk_free_rate, optimization_method):
                        st.balloons()
                        st.success("✅ Оптимизация завершена успешно!")
                    else:
                        st.error("❌ Ошибка при оптимизации портфеля")
                else:
                    st.error("❌ Не удалось загрузить данные для выбранных тикеров")
    
    # Основная область с результатами
    if st.session_state.optimizer.prices is not None and st.session_state.optimizer.optimal_weights is not None:
        optimizer = st.session_state.optimizer
        metrics = optimizer.portfolio_metrics(optimizer.optimal_weights)
        
        # Вкладки с результатами
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Обзор портфеля", 
            "📈 Анализ рисков", 
            "💹 Исторические данные", 
            "🔄 Корреляции",
            "🔍 Детальный анализ"
        ])
        
        with tab1:
            st.subheader("🎯 Оптимальное распределение активов")
            
            # Веса портфеля
            col1, col2 = st.columns([2, 1])
            
            with col1:
                weights_data = pd.DataFrame({
                    'Тикер': optimizer.returns.columns,
                    'Вес (%)': optimizer.optimal_weights * 100
                }).sort_values('Вес (%)', ascending=False)
                
                # Красивая круговая диаграмма
                fig = px.pie(
                    weights_data, 
                    values='Вес (%)', 
                    names='Тикер',
                    title="",
                    color_discrete_sequence=px.colors.sequential.Viridis,
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    showlegend=False,
                    annotations=[dict(text='Распределение', x=0.5, y=0.5, font_size=16, showarrow=False)]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Ключевые метрики")
                
                # Метрики в карточках
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 0.9rem; color: #718096;'>Доходность</div>
                        <div style='font-size: 1.5rem; font-weight: bold; color: #38a169;'>{metrics['annual_return']:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 0.9rem; color: #718096;'>Волатильность</div>
                        <div style='font-size: 1.5rem; font-weight: bold; color: #e53e3e;'>{metrics['annual_volatility']:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 0.9rem; color: #718096;'>Коэф. Шарпа</div>
                        <div style='font-size: 1.5rem; font-weight: bold; color: #3182ce;'>{metrics['sharpe_ratio']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metrics_col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 0.9rem; color: #718096;'>Макс. просадка</div>
                        <div style='font-size: 1.5rem; font-weight: bold; color: #dd6b20;'>{metrics['max_drawdown']:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 0.9rem; color: #718096;'>Коэф. Сортино</div>
                        <div style='font-size: 1.5rem; font-weight: bold; color: #319795;'>{metrics['sortino_ratio']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 0.9rem; color: #718096;'>Бета</div>
                        <div style='font-size: 1.5rem; font-weight: bold; color: #805ad5;'>{metrics['beta']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("🛡️ Анализ рисков")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Распределение доходностей
                portfolio_returns = optimizer.returns.dot(optimizer.optimal_weights)
                fig = px.histogram(
                    x=portfolio_returns * 100,
                    nbins=50,
                    title="📊 Распределение дневных доходностей",
                    labels={'x': 'Доходность (%)', 'y': 'Частота'},
                    color_discrete_sequence=['#667eea']
                )
                fig.add_vline(x=metrics['var_95'] * 100, line_dash="dash", line_color="red", 
                            annotation_text=f"VaR 95%: {metrics['var_95']:.2%}")
                fig.add_vline(x=metrics['cvar_95'] * 100, line_dash="dash", line_color="darkred",
                            annotation_text=f"CVaR 95%: {metrics['cvar_95']:.2%}")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Просадки
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=metrics['drawdown'].index,
                    y=metrics['drawdown'].values * 100,
                    fill='tozeroy',
                    line=dict(color='#e53e3e'),
                    name='Просадка',
                    fillcolor='rgba(229, 62, 62, 0.3)'
                ))
                fig.update_layout(
                    title="📉 История просадок портфеля",
                    xaxis_title="Дата",
                    yaxis_title="Просадка (%)",
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Метрики риска в карточках
                st.subheader("📊 Показатели риска")
                risk_cols = st.columns(2)
                with risk_cols[0]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 0.8rem; color: #718096;'>VaR (95%)</div>
                        <div style='font-size: 1.2rem; font-weight: bold; color: #e53e3e;'>{metrics['var_95']:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 0.8rem; color: #718096;'>Волатильность</div>
                        <div style='font-size: 1.2rem; font-weight: bold; color: #dd6b20;'>{metrics['annual_volatility']:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_cols[1]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 0.8rem; color: #718096;'>CVaR (95%)</div>
                        <div style='font-size: 1.2rem; font-weight: bold; color: #c53030;'>{metrics['cvar_95']:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 0.8rem; color: #718096;'>Макс. просадка</div>
                        <div style='font-size: 1.2rem; font-weight: bold; color: #9c4221;'>{metrics['max_drawdown']:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.subheader("💹 Исторические данные")
            
            display_type = st.radio("**Тип данных:**", ["Цены", "Доходности"], horizontal=True)
            
            if display_type == "Цены":
                data_to_show = optimizer.prices
                title = "📊 Исторические цены акций"
                # Нормализация цен
                normalized_prices = (optimizer.prices / optimizer.prices.iloc[0]) * 100
                fig = px.line(
                    normalized_prices.reset_index().melt(id_vars=['TRADEDATE'], 
                                                         value_name='Цена', 
                                                         var_name='Тикер'),
                    x='TRADEDATE',
                    y='Цена',
                    color='Тикер',
                    title="",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                y_title = "Нормализованная цена (база=100)"
            else:
                data_to_show = optimizer.returns
                title = "📈 Исторические доходности акций"
                fig = px.line(
                    data_to_show.reset_index().melt(id_vars=['TRADEDATE'], 
                                                   value_name='Доходность', 
                                                   var_name='Тикер'),
                    x='TRADEDATE',
                    y='Доходность',
                    color='Тикер',
                    title="",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                y_title = "Доходность"
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Дата",
                yaxis_title=y_title,
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Дополнительные метрики по активам
            st.subheader("📋 Статистика по активам")
            
            assets_stats = []
            for ticker in optimizer.returns.columns:
                asset_returns = optimizer.returns[ticker]
                assets_stats.append({
                    'Тикер': ticker,
                    'Средняя доходность (%)': asset_returns.mean() * 100,
                    'Волатильность (%)': asset_returns.std() * 100,
                    'Мин. доходность (%)': asset_returns.min() * 100,
                    'Макс. доходность (%)': asset_returns.max() * 100,
                    'Вес в портфеле (%)': optimizer.optimal_weights[list(optimizer.returns.columns).index(ticker)] * 100
                })
            
            stats_df = pd.DataFrame(assets_stats)
            styled_df = stats_df.style.format({
                'Средняя доходность (%)': '{:.2f}%',
                'Волатильность (%)': '{:.2f}%',
                'Мин. доходность (%)': '{:.2f}%',
                'Макс. доходность (%)': '{:.2f}%',
                'Вес в портфеле (%)': '{:.2f}%'
            }).background_gradient(subset=['Средняя доходность (%)'], cmap='Greens')\
              .background_gradient(subset=['Вес в портфеле (%)'], cmap='Purples')
            
            st.dataframe(styled_df, use_container_width=True)
        
        with tab4:
            st.subheader("🔄 Корреляционная матрица")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                corr_matrix = optimizer.returns.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="",
                    width=600,
                    height=500
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Анализ диверсификации
                st.subheader("🎯 Анализ диверсификации")
                
                avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                min_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()
                max_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9rem; color: #718096;'>Средняя корреляция</div>
                    <div style='font-size: 1.5rem; font-weight: bold; color: #3182ce;'>{avg_correlation:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9rem; color: #718096;'>Мин. корреляция</div>
                    <div style='font-size: 1.5rem; font-weight: bold; color: #38a169;'>{min_correlation:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9rem; color: #718096;'>Макс. корреляция</div>
                    <div style='font-size: 1.5rem; font-weight: bold; color: #e53e3e;'>{max_correlation:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Оценка диверсификации
                st.subheader("📊 Оценка портфеля")
                if avg_correlation < 0.3:
                    st.success("✅ **Отличная диверсификация** - низкая корреляция между активами")
                elif avg_correlation < 0.6:
                    st.info("ℹ️ **Умеренная диверсификация** - средний уровень корреляции")
                else:
                    st.warning("⚠️ **Низкая диверсификация** - высокие корреляции между активами")
        
        with tab5:
            st.subheader("🔍 Детальный анализ активов")
            
            # Статистика по каждому активу
            st.subheader("📈 Детальная статистика активов")
            
            detailed_stats = []
            for ticker in optimizer.returns.columns:
                asset_returns = optimizer.returns[ticker]
                asset_weights = optimizer.optimal_weights[list(optimizer.returns.columns).index(ticker)]
                
                # Расширенная статистика
                detailed_stats.append({
                    'Тикер': ticker,
                    'Годовая доходность (%)': asset_returns.mean() * 252 * 100,
                    'Годовая волатильность (%)': asset_returns.std() * np.sqrt(252) * 100,
                    'Коэф. Шарпа': (asset_returns.mean() * 252 - risk_free_rate) / (asset_returns.std() * np.sqrt(252)),
                    'Вес в портфеле (%)': asset_weights * 100,
                    'Вклад в риск (%)': asset_weights * asset_returns.std() * np.sqrt(252) * 100,
                    'Skewness': stats.skew(asset_returns),
                    'Kurtosis': stats.kurtosis(asset_returns),
                    'Корреляция с портфелем': np.corrcoef(asset_returns, optimizer.returns.dot(optimizer.optimal_weights))[0, 1]
                })
            
            detailed_df = pd.DataFrame(detailed_stats)
            
            # Стилизация таблицы
            styled_detailed = detailed_df.style.format({
                'Годовая доходность (%)': '{:.2f}%',
                'Годовая волатильность (%)': '{:.2f}%',
                'Коэф. Шарпа': '{:.2f}',
                'Вес в портфеле (%)': '{:.2f}%',
                'Вклад в риск (%)': '{:.2f}%',
                'Skewness': '{:.3f}',
                'Kurtosis': '{:.3f}',
                'Корреляция с портфелем': '{:.3f}'
            }).background_gradient(subset=['Годовая доходность (%)'], cmap='Greens')\
              .background_gradient(subset=['Вес в портфеле (%)'], cmap='Purples')\
              .background_gradient(subset=['Коэф. Шарпа'], cmap='Blues')
            
            st.dataframe(styled_detailed, use_container_width=True)
            
            # Эффективная граница
            st.subheader("📊 Эффективная граница")
            
            # Генерация случайных портфелей
            n_portfolios = 10000
            results = np.zeros((n_portfolios, 3))
            weights_record = []
            
            with st.spinner("⏳ Генерируем эффективную границу..."):
                for i in range(n_portfolios):
                    weights = np.random.random(len(optimizer.returns.columns))
                    weights /= np.sum(weights)
                    weights_record.append(weights)
                    port_metrics = optimizer.portfolio_metrics(weights)
                    results[i, 0] = port_metrics['annual_volatility']
                    results[i, 1] = port_metrics['annual_return']
                    results[i, 2] = port_metrics['sharpe_ratio']
            
            # Визуализация
            fig = go.Figure()
            
            # Случайные портфели
            fig.add_trace(go.Scatter(
                x=results[:, 0], y=results[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=results[:, 2],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Коэф. Шарпа")
                ),
                name='Случайные портфели'
            ))
            
            # Оптимальный портфель
            fig.add_trace(go.Scatter(
                x=[metrics['annual_volatility']],
                y=[metrics['annual_return']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Оптимальный портфель'
            ))
            
            fig.update_layout(
                title="",
                xaxis_title="Волатильность (риск)",
                yaxis_title="Доходность",
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Анализ вклада в риск
            st.subheader("🎯 Вклад активов в риск портфеля")
            
            risk_contribution = []
            for i, ticker in enumerate(optimizer.returns.columns):
                weight = optimizer.optimal_weights[i]
                volatility = optimizer.returns[ticker].std() * np.sqrt(252)
                risk_contribution.append({
                    'Тикер': ticker,
                    'Вес (%)': weight * 100,
                    'Вклад в риск (%)': weight * volatility * 100,
                    'Отношение вклада к весу': (weight * volatility) / weight if weight > 0 else 0
                })
            
            risk_df = pd.DataFrame(risk_contribution)
            
            fig_risk = px.bar(
                risk_df,
                x='Тикер',
                y='Вклад в риск (%)',
                title="",
                color='Вклад в риск (%)',
                color_continuous_scale='Viridis'
            )
            fig_risk.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Тикер",
                yaxis_title="Вклад в риск (%)",
                height=400
            )
            st.plotly_chart(fig_risk, use_container_width=True)
    
    else:
        # Домашняя страница
        st.markdown("""
        <div style='text-align: center; padding: 3rem 1rem;'>
            <h2 style='color: #2d3748; margin-bottom: 1rem;'>🚀 Добро пожаловать в AI Portfolio Optimizer</h2>
            <p style='font-size: 1.3rem; color: #4a5568; line-height: 1.6;'>
                Профессиональный инструмент для интеллектуальной оптимизации <br>инвестиционных портфелей акций Московской биржи
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #2d3748; margin-bottom: 1rem;'>📈 Автоматическая оптимизация</h3>
                <p style='color: #718096; line-height: 1.5;'>Передовые математические алгоритмы для поиска оптимального распределения активов с учетом ваших целей и толерантности к риску</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #2d3748; margin-bottom: 1rem;'>🛡️ Управление рисками</h3>
                <p style='color: #718096; line-height: 1.5;'>Расчет ключевых показателей риска: VaR, CVaR, просадки, волатильность и другие метрики для безопасного инвестирования</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #2d3748; margin-bottom: 1rem;'>📊 Продвинутая аналитика</h3>
                <p style='color: #718096; line-height: 1.5;'>Корреляционный анализ, эффективная граница, детальная статистика и визуализация для принятия взвешенных решений</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Дополнительная информация
        st.markdown("---")
        st.info("💡 **Совет:** Используйте панель слева для настройки параметров портфеля и запуска оптимизации. Выберите готовый тип портфеля или создайте свой собственный!")

if __name__ == "__main__":
    main()
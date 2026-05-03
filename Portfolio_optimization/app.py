import streamlit as st
import os
import numpy as np
from datetime import datetime, timedelta

from src.data_loader import MoexDataLoader
from src.models import PortfolioMath
from src.visuals import Visualizer
from src.expert_engine import ExpertAnalyst

st.set_page_config(page_title="Quant Portfolio AI Pro", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_metric_card(label, value, is_percent=True, invert_color=False):
    formatted_val = f"{value * 100:.2f}%" if is_percent else f"{value:.2f}"
    color_class = "sub-positive" if (value > 0 and not invert_color) or (value < 0 and invert_color) else "sub-negative"
    st.markdown(f"""
        <div class="glass-card metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{formatted_val}</div>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Инициализация хранилища состояний (Session State)
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    st.markdown("""
        <div class="main-header">
            <h1>Quant Portfolio AI <span style="color:#00f2fe;">PRO</span></h1>
            <p>Институциональная платформа машинного обучения и квант-анализа Московской Биржи</p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h2 class='sidebar-title'>Управление портфелем</h2>", unsafe_allow_html=True)
        
        portfolios = {
            "Голубые фишки (Надежность)": "SBER, LKOH, GAZP, GMKN, TATN, ROSN, NVTK",
            "IT-сектор (Технологии)": "YNDX, POSI, ASTR, VKCO, SOFL, HEAD, DIAS",
            "Нефтегазовый сектор": "LKOH, GAZP, ROSN, SNGS, NVTK, TATN, TRNFP",
            "Финансовый сектор": "SBER, VTBR, TCSG, CBOM, BSPB, MOEX",
            "Металлургия и добыча": "GMKN, CHMF, NLMK, MAGN, PLZL, ALRS, UGLD",
            "Потребительский сектор": "MGNT, OZON, FIXP, MVID, BELU, AQUA",
            "Электроэнергетика": "IRAO, HYDR, FEES, UPRO, MSNG",
            "Девелопмент (Недвижимость)": "PIKK, SMLT, LSRG, ETAL",
            "Транспортный сектор": "FLOT, AFLT, FESH, NMTP",
            "Химия и Нефтехимия": "PHOR, AKRN, NKNC, KAZT",
            "Дивидендные аристократы": "LKOH, SBER, TATN, CHMF, PLZL, BSPB",
            "Свой вариант": ""
        }
        
        choice = st.selectbox("Базовый бенчмарк-портфель", list(portfolios.keys()))
        tickers_input = st.text_area("Тикеры активов (через запятую)", value=portfolios[choice], height=100)
        
        st.markdown("---")
        
        opt_strategy = st.selectbox("Математическая модель оптимизации", [
            "1. Max Sharpe Ratio (MPT Марковица)", 
            "2. Risk Parity (Паритет Риска Рей Далио)",
            "3. Global Minimum Volatility (Защитный)",
            "4. Maximum Diversification (Макс. диверсификация)",
            "5. Minimum Tail Risk CVaR (Антикризисный)",
            "6. Hierarchical Risk Parity (Machine Learning)"
        ])
        
        col1, col2 = st.columns(2)
        with col1:
            end_date = datetime.now()
            start_date = st.date_input("Начало", end_date - timedelta(days=365*3))
        with col2:
            rfr = st.number_input("Безриск (%)", value=16.0, step=0.5) / 100
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("ЗАПУСТИТЬ ИИ-АНАЛИЗ", use_container_width=True, type="primary")

    # ЛОГИКА КНОПКИ: Только считаем и сохраняем в память
    if run_btn:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if len(tickers) < 2:
            st.error("Для работы алгоритмов требуется минимум 2 актива.")
            return

        with st.spinner('Соединение с API Московской Биржи (MOEX)...'):
            loader = MoexDataLoader()
            result = loader.fetch_all(tickers, start_date, end_date)
            if result is None:
                st.error("Ошибка загрузки. Проверьте правильность тикеров.")
                return
            prices, returns = result
            
        with st.spinner(f'Вычисление тензоров и оптимизация ({opt_strategy.split("(")[0].strip()})...'):
            math_engine = PortfolioMath(returns, rfr)
            
            if opt_strategy.startswith("1"): weights = math_engine.optimize_sharpe()
            elif opt_strategy.startswith("2"): weights = math_engine.optimize_risk_parity()
            elif opt_strategy.startswith("3"): weights = math_engine.optimize_min_vol()
            elif opt_strategy.startswith("4"): weights = math_engine.optimize_max_div()
            elif opt_strategy.startswith("5"): weights = math_engine.optimize_min_cvar()
            elif opt_strategy.startswith("6"): weights = math_engine.optimize_hrp()
                
            metrics = math_engine.calc_metrics(weights)
            
            # СОХРАНЯЕМ В ПАМЯТЬ СЕССИИ
            st.session_state.prices = prices
            st.session_state.returns = returns
            st.session_state.weights = weights
            st.session_state.metrics = metrics
            st.session_state.analysis_done = True

    # ЛОГИКА ОТРИСОВКИ: Рисуем всегда, если анализ был хотя бы раз выполнен
    if st.session_state.analysis_done:
        # Достаем данные из памяти
        prices = st.session_state.prices
        returns = st.session_state.returns
        weights = st.session_state.weights
        metrics = st.session_state.metrics

        st.markdown("<div class='success-banner'>✅ Квант-оптимизация и бэктестинг успешно завершены</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: render_metric_card("Ожид. Доходность (CAGR)", metrics['return'])
        with c2: render_metric_card("Волатильность (Risk)", metrics['volatility'], invert_color=True)
        with c3: render_metric_card("Коэффициент Шарпа", metrics['sharpe'], is_percent=False)
        with c4: render_metric_card("Max Историч. Просадка", metrics['max_drawdown'], invert_color=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        t1, t2, t3, t4, t5, t6 = st.tabs([
            "Аллокация (ИИ)", 
            "Риски & VaR", 
            "Матрица Корреляций", 
            "Бэктестинг",
            "Графики цен",
            "База данных"
        ])
        
        with t1:
            col_chart, col_ai = st.columns([1.2, 1])
            with col_chart:
                st.plotly_chart(Visualizer.plot_allocation(weights, returns.columns), use_container_width=True)
            with col_ai:
                st.markdown("<h3 style='color: #e2e8f0;'>🤖 ИИ-Аудит Портфеля</h3>", unsafe_allow_html=True)
                report = ExpertAnalyst.generate_report(metrics, returns, weights)
                st.markdown(f'<div class="expert-panel">{report}</div>', unsafe_allow_html=True)
                
        with t2:
            st.markdown("<h3 style='text-align:center; color:#94a3b8;'>Квантильный анализ хвостовых рисков (Fat Tails)</h3>", unsafe_allow_html=True)
            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(Visualizer.plot_distribution_and_qq(metrics['daily_returns']), use_container_width=True)
            with c2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #f59e0b; margin-bottom:15px;">
                    <h4 style="margin:0; color:#fcd34d;">Historical VaR (95%)</h4>
                    <p style="margin-top:5px; color:#cbd5e1;">В 5% худших дней убыток превысит <b>{-np.percentile(metrics['daily_returns'], 5)*100:.2f}%</b></p>
                </div>
                <div class="glass-card" style="border-left: 4px solid #ef4444;">
                    <h4 style="margin:0; color:#fca5a5;">Cornish-Fisher CVaR (95%)</h4>
                    <p style="margin-top:5px; color:#cbd5e1;">Ожидаемый годовой крах при черном лебеде: <b>{-metrics['cvar_95']*100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
        with t3:
            st.plotly_chart(Visualizer.plot_correlation_heatmap(returns), use_container_width=True)
            
        with t4:
            st.plotly_chart(Visualizer.plot_rolling_metrics(metrics['daily_returns']), use_container_width=True)

        with t5:
            st.markdown("<h3 style='text-align:center; color:#94a3b8;'>Детальный анализ стоимости активов</h3>", unsafe_allow_html=True)
            
            st.markdown("#### 🔍 Индивидуальный обзор")
            # Теперь selectbox не будет сбрасывать страницу!
            selected_ticker = st.selectbox("Выберите актив для детального анализа:", prices.columns, index=0)
            st.plotly_chart(
                Visualizer.plot_individual_price(prices, selected_ticker), 
                use_container_width=True,
                config={
                    'scrollZoom': True,  # Включает зум колёсиком мыши
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'zoom2d'],
                    'displaylogo': False
                }
            )
            
            st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)
            
            st.markdown("#### Сравнительная динамика (Все активы)")
            normalize = st.toggle("Нормализовать графики (База = 100 пунктов)", value=True)
            
            if normalize:
                plot_prices = (prices / prices.iloc[0]) * 100
                y_label = "Нормализованная доходность (База=100)"
            else:
                plot_prices = prices
                y_label = "Абсолютная цена закрытия (RUB)"
                
            st.plotly_chart(
                Visualizer.plot_price_history(plot_prices, y_label), 
                use_container_width=True,
                config={
                    'scrollZoom': True,
                    'displayModeBar': True, 
                    'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'zoom2d'],
                    'displaylogo': False
                }
            )
            
        with t6:
            st.markdown("<h3 style='text-align:center; color:#94a3b8;'>Исторические цены закрытия (С учетом сплитов)</h3>", unsafe_allow_html=True)
            display_prices = prices.copy()
            display_prices.index = display_prices.index.strftime('%Y-%m-%d')
            
            c1, c2 = st.columns([4, 1])
            with c1:
                st.dataframe(display_prices, use_container_width=True, height=500)
            with c2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                csv = display_prices.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Скачать датасет (CSV)",
                    data=csv,
                    file_name="moex_adjusted_prices.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
                st.info("**Подсказка:** Данные очищены от сплитов (дробления акций) и выходных дней. Идеально подходят для выгрузки в Excel или Python.")

if __name__ == "__main__":
    main()